"""Distractor Transfer Gap (DTG) diagnostic (CIRO paper, Section 6.2).

DTG is the RL-flavoured analogue of CIR's Interventional Transfer Gap (ITG),
using the Distracting Control Suite's distinct training and evaluation
distractor pools in place of a correlated/uncorrelated split:

    1.  Train a linear probe on representations from the training distractor
        pool, and evaluate it on a held-in split of that same pool ->  score_ID.
    2.  Evaluate the *same* probe on the held-out evaluation-distractor pool
        ->  score_OOD.
    3.  DTG = (score_ID - score_OOD) / score_ID * 100.

A lower DTG indicates the learned features are robust to distractor shift.

`dtg_formula` is a pure function so its arithmetic (and edge cases, e.g. an ID
score of 0) can be unit-tested directly.
"""

from typing import Optional

import numpy as np
import torch

__all__ = ["dtg_formula", "run_dtg"]


def dtg_formula(score_id: float, score_ood: float) -> float:
    """DTG = (score_ID - score_OOD) / score_ID * 100, guarding score_ID <= 0.

    Section 6.2, step 3. If the ID score is not strictly positive the percentage
    drop is undefined -> return NaN instead of a misleading number.

    Args:
        score_id: metric where higher is better (e.g. accuracy) on the ID split.
        score_ood: the same metric on the OOD (eval-distractor) split.

    Returns:
        float percentage drop, or NaN when score_id <= 0.
    """
    if score_id <= 0:
        return float("nan")
    return (score_id - score_ood) / score_id * 100.0


def embed_frames(encode_fn, frames: np.ndarray, device) -> np.ndarray:
    """Embed a (n, H, W, C) float array into (n, d) encoder features.

    Frames are normalised to [0, 1] to match the training pipeline's input
    scaling.
    """
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32))
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    x = x / 255.0
    with torch.no_grad():
        return encode_fn(x.to(device)).cpu().numpy()


def run_dtg(
    encode_fn,
    train_pool_frames: np.ndarray,
    train_pool_labels: np.ndarray,
    eval_pool_frames: np.ndarray,
    eval_pool_labels: np.ndarray,
    val_fraction: float = 0.2,
    device: torch.device = torch.device("cpu"),
    seed: int = 0,
) -> dict:
    """Train a linear probe on the train distractor pool; score ID and OOD.

    The training-distractor-pool frames are split (via `val_fraction`) into a
    probe-training split and a held-in training-distractor ID split. The probe
    is fit on the former and scored on (i) the ID split and (ii) the
    evaluation-distractor pool.

    Args:
        encode_fn: (B, 3, H, W) -> (B, d) trained image encoder.
        train_pool_frames / labels: (n, H, W, C) and (n,) from the training pool.
        eval_pool_frames / labels: (m, H, W, C) and (m,) from the eval pool.
        val_fraction: fraction of the training pool held out as the ID split.
        device: torch device.
        seed: RNG seed for the deterministic ID split.

    Returns:
        dict with keys score_id, score_ood, dtg, n_val, n_train.
    """
    from sklearn.linear_model import LogisticRegression

    feat_train = embed_frames(encode_fn, train_pool_frames, device)
    feat_eval = embed_frames(encode_fn, eval_pool_frames, device)

    rng = np.random.RandomState(seed)
    n = len(feat_train)
    perm = rng.permutation(n)
    n_val = int(val_fraction * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    probe = LogisticRegression(max_iter=1000)
    probe.fit(feat_train[tr_idx], np.asarray(train_pool_labels)[tr_idx])

    score_id = float(probe.score(feat_train[val_idx], np.asarray(train_pool_labels)[val_idx]))
    score_ood = float(probe.score(feat_eval, np.asarray(eval_pool_labels)))

    return {
        "score_id": score_id,
        "score_ood": score_ood,
        "dtg": dtg_formula(score_id, score_ood),
        "n_val": int(n_val),
        "n_train": int(n - n_val),
    }