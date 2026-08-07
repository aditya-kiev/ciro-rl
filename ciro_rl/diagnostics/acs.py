"""Axis-Wise Causal Sensitivity (ACS) diagnostic (CIRO paper, Section 6.1).

ACS measures, for each causal latent i in S, how concentrated its causal influence
is on a single representation coordinate after propagating an intervention through
the transition dynamics for h steps.

Protocol (Section 6.1, matched exactly):
    For each causal latent i in S:
        1.  z_t ~ p(z_t) (n samples) via `sample_states_and_actions`.
        2.  z'_t = do(z_{t,i} := delta)            (hard one-time intervention).
        3.  Propagate z_t and z'_t for h steps under the same held-fixed action
            sequence and the same shared noise stream, giving z_{t+h}, z'_{t+h}.
        4.  Render and encode both: h = enc(render(z_{t+h})), h' = enc(render(z'_{t+h})).
        5.  S[i, j] = (1/delta) * E[ |h'_j - h_j| ].
    Then Hungarian-match causal latents to coordinates on cost = -S, compute
    concentration conc_i = S[i, j*(i)] / sum_j S[i, j], and report
    ACS = (1/|S|) * sum_i conc_i in [0, 1].

The encode_fn is a plain callable (B, 3, H, W) -> (B, d), so it can be either a
trained CNN encoder (nn.Module, e.g. model.query) or a tiny linear map injected
by a unit test. The SCM interface is duck-typed.
"""

from typing import Callable, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def _encode_in_chunks(encode_fn: Callable, frames: torch.Tensor, batch_size: int, device) -> torch.Tensor:
    """Encode large batches in chunks to bound memory; returns (n, d)."""
    outs = []
    n = frames.shape[0]
    for s in range(0, n, batch_size):
        with torch.no_grad():
            outs.append(encode_fn(frames[s : s + batch_size].to(device)))
    return torch.cat(outs, dim=0)


def sensitivity_matrix(
    scm,
    encode_fn: Callable,
    n_samples: int = 512,
    delta: float = 1.0,
    h: int = 0,
    batch_size: int = 64,
    causal_latents: Optional[Sequence[int]] = None,
    device=None,
) -> np.ndarray:
    """Return the (|S|, d) sensitivity matrix (ACS step 5).

    Matches S[i, j] = (1/delta) * E|h'_j - h_j|, with h' from the intervened then
    rolled-forward state and h from the baseline. The shared noise stream makes
    the two rollouts comparable.
    """
    causal = list(causal_latents) if causal_latents is not None else scm.get_causal_latent_indices()
    z0, actions = scm.sample_states_and_actions(n_samples, h, device)
    noise = scm.new_noise(n_samples, h, device)
    with torch.no_grad():
        z_ref = scm.rollout(z0, actions, noise)
        x_ref = scm.render(z_ref)
        f_ref = _encode_in_chunks(encode_fn, x_ref, batch_size, device)

    rows = []
    for latent_idx in causal:
        zi = scm.intervene(z0, latent_idx, delta)
        with torch.no_grad():
            z_int = scm.rollout(zi, actions, noise)
            x_int = scm.render(z_int)
            f_int = _encode_in_chunks(encode_fn, x_int, batch_size, device)
        row = (f_int - f_ref).abs().mean(dim=0) / delta  # (d,)
        rows.append(row.cpu().numpy())
    return np.array(rows)  # (n_causal, d)


def hungarian_match(S: np.ndarray) -> np.ndarray:
    """Max-weight assignment of causal-latent rows to coordinates (ACS step).

    Maximises total matched sensitivity on cost matrix -S. Returns an int array
    mapping causal-latent index i -> matched coordinate (never revisits a coord).
    """
    S = np.asarray(S, dtype=np.float64)
    rows, cols = linear_sum_assignment(-S)
    mapping = np.full(S.shape[0], -1, dtype=int)
    mapping[rows] = cols
    return mapping


def concentration(S: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """conc_i = S[i, j*(i)] / sum_j S[i, j]  (conc_i per causal latent)."""
    S = np.asarray(S, dtype=np.float64)
    row_sum = np.where(S.sum(1) > 0, S.sum(1), 1.0)
    scores = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        if mapping[i] >= 0:
            scores[i] = S[i, mapping[i]] / row_sum[i]
    return scores


def average_causal_sensitivity(
    scm,
    encode_fn: Callable,
    n_samples: int = 512,
    delta: float = 1.0,
    h: int = 0,
    batch_size: int = 64,
    device=None,
) -> dict:
    """Full ACS diagnostic for a trained encoder.

    Args:
        scm: SCM-MDP object (duck-typed): get_causal_latent_indices,
            sample_states_and_actions, new_noise, rollout, intervene, render.
        encode_fn: (B, 3, H, W) -> (B, d) image encoder to evaluate.
        n_samples: reference batch size.
        delta: intervention magnitude (Section 6.1).
        h: rollout horizon (h == 0 is the static diagnostic).
        batch_size: encoding chunk size.
        device: torch device (default cpu).

    Returns:
        dict with keys 'acs', 'sensitivity_matrix' ((n_causal, d) numpy),
        'matching' (list of (latent_idx, coord_idx)), 'concentration' (array).
    """
    S = sensitivity_matrix(scm, encode_fn, n_samples, delta, h, batch_size,
                           None, device)
    mapping = hungarian_match(S)
    conc = concentration(S, mapping)
    matched = int((mapping >= 0).sum())
    acs = float(conc[mapping >= 0].mean()) if matched > 0 else 0.0
    return {
        "acs": acs,
        "sensitivity_matrix": S,
        "matching": list(zip(np.where(mapping >= 0)[0].tolist(), mapping[mapping >= 0].tolist())),
        "concentration": conc,
    }


__all__ = ["sensitivity_matrix", "hungarian_match", "concentration",
           "average_causal_sensitivity"]