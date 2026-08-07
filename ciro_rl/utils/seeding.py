"""Reproducibility: uniform seeding across Python, NumPy, and PyTorch/CUDA.

Matches the CIRO paper's Reproducibility Statement: every experiment is seeded
with a single integer `seed`, and the default 3 seeds used everywhere in this
repo are 0, 1, 2 (the same convention as the CIR paper).
"""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch/CUDA from a single integer.

    Args:
        seed: Master seed. The experiment scripts pass one value per run
            (defaults: 0, 1, 2).
        deterministic: Enable deterministic CuDNN; slightly slower but
            reproducible. Turn off only if a numerically-better non-reproducible
            algorithm is required.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


DEFAULT_SEEDS = (0, 1, 2)


def default_seeds() -> tuple:
    """The 3 seeds used by default across the repo (matches the CIR paper)."""
    return DEFAULT_SEEDS


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve 'cuda' | 'cpu' | 'auto' to a concrete torch.device."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)