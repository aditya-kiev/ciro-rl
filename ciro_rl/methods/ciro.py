"""CIRO: full method. CURL contrastive objective + residual-then-per-coordinate
HSIC penalty on the query/key projection pair.

Mirrors causal-cir's `losses/cir.py` structure exactly, with the difference that
the residual construction is applied to CURL's query/key pair (per the CIRO paper
Algorithm 1) rather than to two symmetric SimCLR views.

Algorithm 1 of the CIRO paper (anchors in brackets):
    2  hq, hk  <- proj(enc(request)), proj(enc(key))            (line 2)
    3  Hbar     <- (hq + hk)/2                                    (line 3)
    4  rq <- hq - Hbar,  rk <- hk - Hbar                          (line 4)
    5  R <- concat(rq, rk)                                        (line 5)
   11  R_CIRO = (1/n^2) sum_{j!=k} hat{HSIC}(R_{:,j}, R_{:,k})     (line 11)
   12  L_CURL                                                     (line 12)
   13  L = L_CURL + lambda * R_CIRO                                (line 13)
The per-coordinate kernels are RBF with median-heuristic bandwidth; the training
path uses the RFF approximation (App. A), the test path the exact V-statistic.

Args about which representations the penalty is applied to (`placement`):
  - 'encoder': penalty applied to the encoder outputs h (pre-projection).
  - 'projection': penalty applied to the projected embeddings z.
The default is 'encoder' (CURL has no projection head by default, App. C.1) and
the placement is ablated in the CIRO paper Table 2.
"""

from typing import Optional

import torch

from ..utils.hsic import pairwise_hsic_exact, pairwise_hsic_rff
from .curl import curl_contrastive_loss


class CIROLoss(torch.nn.Module):
    """L = L_CURL + lambda * R_CI O(residuals of query/key), placement-selectable.

    Args:
        lambda_: penalty weight (default 0.05, matching the CIR paper).
        sigma: RBF bandwidth; None = median heuristic.
        hsic_mode: 'exact' (used by unit tests) or 'rff' (used for training).
        rff_dim: random features per coordinate (default 128).
    """

    def __init__(self, lambda_: float = 0.05, sigma: Optional[float] = None,
                 hsic_mode: str = "rff", rff_dim: int = 128):
        super().__init__()
        self.lambda_ = lambda_
        self.sigma = sigma
        self.hsic_mode = hsic_mode
        self.rff_dim = rff_dim

    def _penalty(self, R: torch.Tensor) -> torch.Tensor:
        if self.hsic_mode == "exact":
            return pairwise_hsic_exact(R, self.sigma)
        return pairwise_hsic_rff(R, self.sigma, self.rff_dim)

    def residuals(self, hq: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
        """Per-pair mean subtraction -> stacked residual matrix R (2N, d).

        Algorithm 1 lines 3-5. Returns an (2N, d) matrix.
        """
        h_bar = (hq + hk) / 2.0
        rq = hq - h_bar
        rk = hk - h_bar
        return torch.cat([rq, rk], dim=0)

    def forward(
        self,
        hq: torch.Tensor,
        hk: torch.Tensor,
        zq: torch.Tensor,
        zk: torch.Tensor,
        W: torch.Tensor,
        placement: str = "encoder",
    ) -> dict:
        """Compute L_CURL + lambda * R_CIO.

        Args:
            hq, hk: encoder outputs (query, key) (N, d).
            zq, zk: projection-head outputs (query, key) (N, d) (== h when no proj).
            W: bilinear matrix used by the contrastive term.
            placement: 'encoder' -> penalize h; 'projection' -> penalize z.

        Returns:
            dict of named loss components (all scalar tensors).
        """
        a, b = (hq, hk) if placement == "encoder" else (zq, zk)
        R = self.residuals(a, b)
        r_cio = self._penalty(R)
        c = curl_contrastive_loss(zq, zk, W)
        total = c + self.lambda_ * r_cio
        return {"contrastive": c, "ciro": r_cio, "residual_mat": R, "total": total}