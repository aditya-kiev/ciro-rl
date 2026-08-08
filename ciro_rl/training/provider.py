"""Data providers that turn an SCM-MDP or the Distracting Control Suite into a
stream of (xq, xk, labels) contrastive-training batches.

Positive pairs follow Assumption 1 of the CIRO paper:
  * SCM-MDP : temporal-neighbour pairs within a window |dt| <= w of a single
    logged (offline) trajectory (case (i)). The two views approximately preserve
    the causal parents.
  * DCS     : two distractor/camera-augmented renderings of the same underlying
    state, drawn from the training distractor pool (case (ii)). DCS is
    environment-gated (requires dm_control/MuJoCo).
"""

from typing import Iterable, Optional, Tuple

import numpy as np
import torch

from ..envs.dcs_wrapper import DCSWrapper, DCSError, is_available, sample_pool_frames
from ..envs.scm_mdp import get_scm_mdp


class SMTemporalPairs:
    """Offline temporal-neighbour pairs for a synthetic SCM-MDP.

    Builds a deterministic offline trajectory buffer once (latents, actions,
    rewards) and streams rendered (query, key, reward) pixel pairs, where key is
    a temporal neighbour of query within `window` steps (Assumption 1 case (i)).

    Args:
        kind: 'independent' | 'causal_chain' | 'confounded'.
        window: max |dt| for temporal-neighbour selection.
        n_episodes / episode_len: offline buffer shape.
        seed: determinism.
        device: device for rendered tensors.
    """

    def __init__(
        self,
        kind: str,
        window: int = 4,
        n_episodes: int = 24,
        episode_len: int = 120,
        seed: int = 0,
        latent_dim: int = 8,
        causal_parents=None,
        confounded_pair=(0, 1),
        beta: float = 0.5,
        img_size: int = 32,
        action_dim: int = 1,
        device=None,
    ):
        self.scm = get_scm_mdp(
            kind,
            latent_dim=latent_dim,
            causal_parents=causal_parents,
            confounded_pair=confounded_pair,
            beta=beta,
            img_size=img_size,
            action_dim=action_dim,
            seed=seed,
        )
        self.window = max(1, int(window))
        self.n_episodes = n_episodes
        self.episode_len = episode_len
        self.device = device or torch.device("cpu")
        self.latents, self.actions, self.rewards = self.scm.build_offline_trajectories(
            n_episodes, episode_len, seed
        )
        self._neighbour_offsets = np.concatenate(
            [np.arange(-self.window, 0), np.arange(1, self.window + 1)]
        )

    def _sample_times(self, n: int, rng: np.random.RandomState):
        eps = rng.randint(0, self.n_episodes, size=n)
        ts = rng.randint(0, self.episode_len, size=n)
        dts = rng.choice(self._neighbour_offsets, size=n)
        return eps, ts, dts

    @torch.no_grad()
    def pairs(self, n: int, rng: np.random.RandomState):
        """Return (xq, xk, labels) of size n as torch tensors (queried view first)."""
        eps, ts, dts = self._sample_times(n, rng)
        ts_k = np.clip(ts + dts, 0, self.episode_len - 1)
        zq = torch.as_tensor(self.latents[eps, ts], dtype=torch.float32)
        zk = torch.as_tensor(self.latents[eps, ts_k], dtype=torch.float32)
        labels = torch.as_tensor(self.rewards[eps, ts], dtype=torch.float32)
        xq = self.scm.render(zq)
        xk = self.scm.render(zk)
        return xq, xk, labels

    def batches(self, batch_size: int, steps: int, seed: int = 0) -> Iterable:
        """Yield (xq, xk, labels) tensors, `steps` batches of `batch_size`."""
        rng = np.random.RandomState(seed)
        for _ in range(steps):
            xq, xk, lab = self.pairs(batch_size, rng)
            yield xq.to(self.device), xk.to(self.device), lab.to(self.device)


class DcsPairs:
    """Distractor-augmented contrastive pairs for a Distracting Control Suite task.

    Environment-gated: constructing ``DcsPairs`` raises :class:`DCSError` when
    the dm_control / distracting_control stack is unavailable.
    """

    def __init__(
        self,
        domain: str,
        task: str,
        resolution: int = 64,
        n_frames: int = 16,
        seed: int = 0,
        label_bins: int = 2,
        difficulty: str = "medium",
        device=None,
    ):
        if not is_available():
            raise DCSError(
                "DCS not available; cannot build DcsPairs. Run the SCM-only suites."
            )
        self.domain = domain
        self.task = task
        self.device = device or torch.device("cpu")
        self.wrapper = DCSWrapper(
            domain, task,
            resolution=resolution,
            seed=seed,
            difficulty=difficulty,
        )
        self.n_frames = max(int(n_frames), 8)
        self.label_bins = label_bins
        self.cache = {}

    def load_frames(self, pool: str, n: int, action_seed: int = 0):
        """Load (frames, labels) from a pool, cached per (pool, action_seed)."""
        key = (pool, action_seed)
        if key not in self.cache:
            frames, rewards = sample_pool_frames(self.wrapper, pool, n, action_seed)
            labels = (np.asarray(rewards) > 0.05).astype(int)
            self.cache[key] = (frames, labels)
        return self.cache[key]

    def batches(self, batch_size: int, steps: int, pool: str = "train", seed: int = 0) -> Iterable:
        frames, labels = self.load_frames(pool, max(batch_size * 8, self.n_frames))
        rng = np.random.RandomState(seed)
        for _ in range(steps):
            idx = rng.randint(0, len(frames), size=batch_size)
            xq = _augment(np.asarray(frames)[idx])
            xk = _augment(np.asarray(frames)[idx])
            lab = torch.as_tensor(labels[idx], dtype=torch.long)
            yield torch.from_numpy(xq).to(self.device), torch.from_numpy(xk).to(self.device), lab.to(self.device)


def _augment(frames: np.ndarray) -> np.ndarray:
    """Light augmentation to create two distractor views (flip + small noise)."""
    aug = np.asarray(frames, dtype=np.float32) / 255.0  # (n, H, W, 3)
    rng = np.random.RandomState(np.random.randint(0, 2**31))
    if rng.rand() < 0.5:
        aug = aug[..., ::-1, :]
    aug = np.clip(aug + 0.1 * rng.randn(*aug.shape).astype(np.float32), 0.0, 1.0)
    return np.transpose(aug, (0, 3, 1, 2))  # -> (n, 3, H, W)