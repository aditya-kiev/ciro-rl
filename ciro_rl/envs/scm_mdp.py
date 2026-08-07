"""Synthetic SCM-MDP benchmark generator (CIRO paper, Appendix B.1).

This extends the CIR paper's fixed-random-decoder renderer (the same MLP used in
causal-cir's `data/synthetic_scm.py`: Linear(8,64)->ReLU->Linear(64,128)->ReLU->
Linear(128,3*32*32)->Sigmoid, fixed random init, one seed per benchmark) with a
*transition* function z_{t+1} = f(z_t, a_t, eps) and a causal-parent dependent
reward y_t = h(z_{t,S}, eps_y).

Three benchmarks, matching the CIRO paper's Dataset A / B / C:

  (A) Independent  : every latent follows its own AR(1); causal-parent latents
       additionally receive an action-modulated drift beta * a_t.
  (B) Causal chain : a within-timestep cross-sectional chain coupling persisted
       through time (CIR's static chain coupling carried over) plus the same
       action drift on causal parents S.
  (C) Confounded   : the (A) dynamics except a designated latent pair (j,k) is
       overwritten each step via a shared, temporally-persistent confounder
       u_t = 0.9 u_{t-1} + 0.3 w_t.

Determinism / shared noise. ACS compares an original rollout against a
do-intervened rollout under the *same* dynamics, actions and noise (CIRO .6
step 3). To keep that comparison valid, every exogenous draw (the latent noise
`eps` and, for (C), the confounder innovations `w`) is produced once by
`new_noise()` and threaded through `rollout()`; the noise does not depend on the
state, so the original and intervened rollouts receive identical noise and only
the intervention differs.

Explicit code-level config values (the prose in B.1 has an open TODO for
exactly these; code must run so they live in the config YAML and are
inspectable): causal_parents S = [0,1,2,3], nuisance = [4,5,6,7],
confounder_pair (j,k) = (0,1) for Dataset C, beta = 0.5.
"""

from typing import List, Optional, Tuple

import torch


def _init_decoder(latent_dim: int, img_size: int, seed: int) -> torch.nn.Module:
    """Fixed random MLP decoder, one instantiation per benchmark, never trained."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(latent_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 3 * img_size * img_size),
        torch.nn.Sigmoid(),
    )
    with torch.no_grad():
        net(torch.randn(1, latent_dim, generator=g))
    return net


class NoiseStream:
    """A pre-drawn exogenous-noise stream shared between paired rollouts.

    Holds the per-step latent innovation `eps` (n, h, latent) and, for the
    confounded benchmark, the confounder innovations `w` (n, h+1).
    """

    def __init__(self, eps: torch.Tensor, w: Optional[torch.Tensor] = None):
        self.eps = eps
        self.w = w


class _SCMMDP:
    """Base SCM-MDP generator."""

    def __init__(
        self,
        kind: str,
        latent_dim: int = 8,
        causal_parents: Optional[List[int]] = None,
        nuisance_indices: Optional[List[int]] = None,
        confounded_pair: Tuple[int, int] = (0, 1),
        beta: float = 0.5,
        img_size: int = 32,
        seed: int = 7,
        action_dim: int = 1,
    ):
        self.kind = kind
        self.latent_dim = latent_dim
        self.causal_parents = list(causal_parents if causal_parents is not None else [0, 1, 2, 3])
        self.nuisance_indices = list(
            nuisance_indices if nuisance_indices is not None else [4, 5, 6, 7]
        )
        assert len(self.causal_parents) + len(self.nuisance_indices) == latent_dim
        assert len(set(self.causal_parents)) == len(self.causal_parents)
        self.confounded_pair = tuple(confounded_pair)
        self.beta = beta
        self.img_size = img_size
        self.seed = seed
        self.action_dim = action_dim
        self._decoder = _init_decoder(latent_dim, img_size, seed)
        self._persist = 0.9
        self._innov = 0.3
        self._chain_coeff = 0.8
        self._w_innov = 0.3

    # -- simple accessors ------------------------------------------------
    def get_causal_latent_indices(self) -> List[int]:
        return list(self.causal_parents)

    def get_nuisance_latent_indices(self) -> List[int]:
        return list(self.nuisance_indices)

    # -- sampling ----------------------------------------------------------
    def sample_prior(self, n: int, device=torch.device("cpu")) -> torch.Tensor:
        raise NotImplementedError

    def _draw_actions(self, n, h, device) -> torch.Tensor:
        """Behavior policy: iid U(-1, 1) per (sample, step)."""
        return torch.rand(n, h, self.action_dim, device=device) * 2.0 - 1.0

    def new_noise(self, n: int, h: int, device=torch.device("cpu")) -> NoiseStream:
        """Draw one shared noise sequence for an (original vs intervened) pair."""
        dt_g = torch.Generator().manual_seed((self.seed + 17) * 31)
        eps = torch.randn(n, h, self.latent_dim, generator=dt_g)
        w = None
        if self.kind == "confounded":
            w = torch.randn(n, h + 1, generator=dt_g)
        return NoiseStream(eps, w)

    @torch.no_grad()
    def sample_states_and_actions(
        self, n: int, h: int, device=torch.device("cpu")
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make a (z0, action-window) draw used by ACS.

        Epub returns (z (n, latent), actions (n, h, action_dim)). h == 0 gives
        the static diagnostic (CIO6 h=0 recovers CIR's static ACS).
        """
        actions = self._draw_actions(n, h, device)
        return self.sample_prior(n, device), actions

    # -- deterministic dynamics -----------------------------------------
    def _step(self, z: torch.Tensor, a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """z_{t+1} = f(z_t, a_t, eps). Subclass-specific."""
        raise NotImplementedError

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, actions: torch.Tensor, noise: NoiseStream) -> torch.Tensor:
        """Propagate z0 for len(actions) steps under shared noise.

        z0: (n, latent); actions: (n, h, action_dim); noise from `new_noise`.
        Returns z_{t+h}: (n, latent).
        """
        n, T, _ = actions.shape
        z = z0
        if self.kind == "confounded":
            j, k = self.confounded_pair
            u = torch.zeros(n, device=z0.device)
            w = noise.w  # (n, T+1)
            for t in range(T):
                u = self._persist * u + self._w_innov * w[:, t]
                z_next = self._persist * z + self._innov * noise.eps[:, t]
                z_next[:, self.causal_parents] += self.beta * actions[:, t]
                z_next[:, j] = u + self._innov * noise.eps[:, t, j]
                z_next[:, k] = u + self._innov * noise.eps[:, t, k]
                z = z_next
        else:
            for t in range(T):
                z = self._step(z, actions[:, t], noise.eps[:, t])
        return z

    # -- render / reward ---------------------------------------------------
    def render(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent) -> (B, 3, H, W)."""
        raw = self._decoder(z.float())
        return raw.view(-1, 3, self.img_size, self.img_size)

    def reward(self, z: torch.Tensor, eps_scale: float = 0.1,
               eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """y_t = h(z_{t,S}, eps_y): mean of causal parents + small reward noise.

        `eps` (B,) may be supplied by the caller (e.g. the offline buffer build)
        to keep the trajectory reward deterministic in its seed; otherwise a draw
        from the global torch RNG is used.
        """
        base = z[:, self.causal_parents].mean(dim=1)
        if eps is not None:
            return base + eps_scale * eps
        return base + eps_scale * torch.randn_like(base)

    # -- interventions (CIR Appendix B.1 / ACS API) ------------------------
    def intervene(self, z: torch.Tensor, latent_idx: int, value: float) -> torch.Tensor:
        """do(z_{t,i} := value): hard one-time intervention, other coords same."""
        zi = z.clone()
        zi[:, latent_idx] = value
        return zi

    # -- offline replay buffer -----------------------------------------------
    @torch.no_grad()
    def build_offline_trajectories(
        self, n_episodes: int, ep_len: int, seed: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a fixed offline replay buffer (latents, actions, rewards).

        (latents) (n_episodes, ep_len, latent); (actions) (n_episodes, ep_len, A);
        (rewards) (n_episodes, ep_len). Deterministic in `seed`.
        """
        gs = torch.Generator().manual_seed(seed + 1)
        latents = torch.zeros(n_episodes, ep_len, self.latent_dim)
        latents[:, 0] = torch.randn(n_episodes, self.latent_dim, generator=torch.Generator().manual_seed(seed))
        actions = torch.rand(n_episodes, ep_len, self.action_dim, generator=gs) * 2.0 - 1.0

        if self.kind == "confounded":
            j, k = self.confounded_pair
            u = torch.zeros(n_episodes)
            for t in range(ep_len - 1):
                u = self._persist * u + self._w_innov * torch.randn(n_episodes, generator=gs)
                z = self._persist * latents[:, t] + self._innov * torch.randn(n_episodes, self.latent_dim, generator=gs)
                z[:, self.causal_parents] += self.beta * actions[:, t]
                z[:, j] = u + self._innov * torch.randn(n_episodes, generator=gs)
                z[:, k] = u + self._innov * torch.randn(n_episodes, generator=gs)
                latents[:, t + 1] = z
        else:
            for t in range(ep_len - 1):
                a = actions[:, t]
                z = latents[:, t]
                eps = torch.randn(n_episodes, self.latent_dim, generator=gs)
                latents[:, t + 1] = self._step(z, a, eps)

        rewards = self.reward(
            latents.view(-1, self.latent_dim),
            eps=torch.randn(n_episodes * ep_len, generator=gs),
        ).view(n_episodes, ep_len)
        return latents, actions, rewards


class IndependentSCMMDP(_SCMMDP):
    """(A) per-latent AR(1); causal parents get drift beta * a_t."""

    def sample_prior(self, n, device=torch.device("cpu")):
        return torch.randn(n, self.latent_dim, device=device)

    def _step(self, z, a, eps):
        z_next = self._persist * z + self._innov * eps
        z_next[:, self.causal_parents] += self.beta * a
        return z_next


class CausalChainSCMMDP(_SCMMDP):
    """(B) within-step chain coupling + AR persistence + action drift on S."""

    def sample_prior(self, n, device=torch.device("cpu")):
        z = torch.randn(n, self.latent_dim, device=device)
        for i in range(1, self.latent_dim):
            z[:, i] = self._chain_coeff * z[:, i - 1] + 0.6 * torch.randn(n, device=device)
        return z

    def _step(self, z, a, eps):
        z_next = torch.empty_like(z)
        z_next[:, 0] = self._persist * z[:, 0] + 0.3 * eps[:, 0]
        for i in range(1, self.latent_dim):
            z_next[:, i] = (
                self._chain_coeff * z[:, i - 1] + self._persist * z[:, i] + 0.6 * eps[:, i]
            )
        z_next[:, self.causal_parents] += self.beta * a
        return z_next


class ConfoundedSCMMDP(_SCMMDP):
    """(C) independent except a shared, persistent confounder drives (j, k)."""

    def sample_prior(self, n, device=torch.device("cpu")):
        z = torch.randn(n, self.latent_dim, device=device)
        u = torch.randn(n, device=device)
        j, k = self.confounded_pair
        z[:, j] = u + 0.3 * torch.randn(n, device=device)
        z[:, k] = u + 0.3 * torch.randn(n, device=device)
        return z


_KIND_TO_CLASS = {
    "independent": IndependentSCMMDP,
    "causal_chain": CausalChainSCMMDP,
    "confounded": ConfoundedSCMMDP,
}


def get_scm_mdp(
    kind: str,
    latent_dim: int = 8,
    causal_parents: Optional[List[int]] = None,
    nuisance_indices: Optional[List[int]] = None,
    confounded_pair: Tuple[int, int] = (0, 1),
    beta: float = 0.5,
    img_size: int = 32,
    seed: int = 7,
    action_dim: int = 1,
) -> _SCMMDP:
    """Factory for SCM-MDP benchmarks."""
    if kind not in _KIND_TO_CLASS:
        raise ValueError(f"Unknown SCM kind '{kind}'. Choose one of {list(_KIND_TO_CLASS)}")
    return _KIND_TO_CLASS[kind](
        kind=kind,
        latent_dim=latent_dim,
        causal_parents=causal_parents,
        nuisance_indices=nuisance_indices,
        confounded_pair=confounded_pair,
        beta=beta,
        img_size=img_size,
        seed=seed,
        action_dim=action_dim,
    )