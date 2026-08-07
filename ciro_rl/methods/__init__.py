"""Method-resolution helpers for the experiment entry points.

Exports ``build_model`` (the shared CURL-style encoder model) and
``make_loss_fn`` (method -> loss module, or None for the plain CURL baseline).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .ciro import CIROLoss
from .curl import CURLModel, curl_contrastive_loss  # noqa: F401
from .marginal_hsic import MarginalHSICLoss  # noqa: F401

METHODS = ("curl", "marginal_hsic", "ciro")


def make_loss_fn(method: str, cfg, placement: Optional[str] = "encoder") -> Optional[nn.Module]:
    """Build the method loss (``None`` for the untruse curl baseline).

    Args:
        method:   'curl' | 'marginal_hsic' | 'ciro'.
        cfg:      config mapping (attribute / dict).
        placement: CIRO penalty placement ('encoder' | 'projection').
    """
    lam = float(cfg.get("hsic_lambda", 0.05))
    mode = cfg.get("hsic_mode", "rff")
    rff = int(cfg.get("rff_dim", 128))
    sigma = cfg.get("hsic_sigma", None)
    if method == "curl":
        return None
    if method == "marginal_hsic":
        return MarginalHSICLoss(lambda_=lam, sigma=sigma, hsic_mode=mode, rff_dim=rff)
    if method == "ciro":
        loss = CIROLoss(lambda_=lam, sigma=sigma, hsic_mode=mode, rff_dim=rff)
        loss.placement = placement or "encoder"
        return loss
    raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}")


def build_model(cfg, device: torch.device) -> CURLModel:
    """Build the shared CURL-style encoder model on ``device``."""
    model = CURLModel(
        d_rep=int(cfg.get("d_rep", 64)),
        proj_hidden=int(cfg.get("proj_hidden", 64)),
        proj_out=int(cfg.get("proj_out", 64)),
        proj_enabled=bool(cfg.get("proj_enabled", False)),
    ).to(device)
    return model