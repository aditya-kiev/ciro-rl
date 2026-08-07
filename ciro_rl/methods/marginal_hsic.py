"""Marginal-HSIC baseline: CURL + an HSIC penalty on *raw* (un-residualised)
features of each branch independently, no per-pair mean subtraction.

This is the RL analogue of causal-cir's "SimCLR + HSIC (marginal)". It is the
ablation baseline described in the CIRO paper, Section 7.1: "...a marginal-HSIC
penalty applied to each branch independently".

Penalty: per HMR, the pairwise-coordinate HSIC on the query batch and on the key
batch (each treated as its own (N, d) matrix), summed over both branches. Note
that unlike CIRO there is NO residual construction.
"""

from typing import Optional

import torch

from ..utils.hsic import pairwise_hsic_exact, pairwise_hsic_rff
from .curl import curl_contrastive_loss


class MarginalHSICLoss(torch.nn.Module):
    """L = L_CURL + lambda * (HSIC_pairwise(zq) + HSIC_pairwise(zk)).

    Args:
        lambda_: penalty weight (default 0.05).
        sigma: RBF bandwidth; None -> median heuristic. Only relevant for the
            exact estimator; the RFF mode resolves a per-batch median.
        hsic_mode: 'exact' | 'rff'.
        rff_dim: random features per coordinate (RFF mode only).
    """

    def __init__(self, lambda_: float = 0.05, sigma: Optional[float] = None,
                 hsic_mode: str = "rff", rff_dim: int = 128):
        super().__init__()
        self.lambda_ = lambda_
        self.sigma = sigma
        self.hsic_mode = hsic_mode
        self.rff_dim = rff_dim

    def forward(self, zq: torch.Tensor, zk: torch.Tensor, W: torch.Tensor) -> dict:
        c_loss = curl_contrastive_loss(zq, zk, W)
        if self.hsic_mode == "exact":
            h1 = pairwise_hsic_exact(zq, self.sigma)
            h2 = pairwise_hsic_exact(zk, self.sigma)
        else:
            h1 = pairwise_hsic_rff(zq, self.sigma, self.rff_dim)
            h2 = pairwise_hsic_rff(zk, self.sigma, self.rff_dim)
        r = h1 + h2
        total = c_loss + self.lambda_ * r
        return {
            "contrastive": c_loss,
            "hsic": r,
            "h1": h1,
            "h2": h2,
            "total": total,
        }