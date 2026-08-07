"""CURL baseline: plain bilinear contrastive objective, no regularizer.

Implements the query/momentum-key encoder and the learned bilinear similarity
matrix W that every method in this repo shares. The CIRO paper's Section 4
analyses exactly this objective; Algorithm 1's line 12 defines the loss:

    L_CURL = -(1/N) sum_i log[ exp(hq_i^T W hk_i) / sum_j exp(hq_i^T W hk_j) ]

No projection head is added by default (App. C.1), matching CURL's own design;
the *_proj heads exist only for the placement ablation in Experiments/Table 2.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.cnn_backbone import CNNBackbone
from ..encoders.projection_head import ProjectionHead


def curl_contrastive_loss(hq: torch.Tensor, hk: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """CURL bilinear InfoNCE loss (CIRO paper eq. (1) / Algorithm 1, line 12).

    logits[i, j] = hq_i^T W hk_j; positives are diagonal entries. Implemented
    directly and this is the bilinear form (not cosine similarity) that the paper
    studies.

    Args:
        hq: query projections (N, d).
        hk: key projections (N, d).
        W: bilinear matrix (d, d).

    Returns:
        Scalar L_C in R.
    """
    logits = hq @ W @ hk.T  # (N, N)
    targets = torch.arange(hq.shape[0], device=logits.device)
    # For a minibatch the positives are the self pairs on the diagonal at
    # temperature tau=1.0.
    return F.cross_entropy(logits, targets)


class CURLModel(nn.Module):
    """Query encoder + momentum key encoder + optional projections + bilinear W.

    Args:
        d_rep: encoder representation/size (also the embedding if no projection).
        proj_hidden, proj_out: projection-head sizes (used only when proj_enabled).
        proj_enabled: whether to add a projection head (default False, per CURL).
    """

    def __init__(self, d_rep: int = 64, proj_hidden: int = 64, proj_out: int = 64,
                 proj_enabled: bool = False):
        super().__init__()
        self.query = CNNBackbone(d_rep=d_rep)
        self.key = CNNBackbone(d_rep=d_rep)
        self.proj_enabled = proj_enabled
        self.out_dim = proj_out if proj_enabled else d_rep
        if proj_enabled:
            self.query_proj = ProjectionHead(d_rep, proj_hidden, proj_out)
            self.key_proj = ProjectionHead(d_rep, proj_hidden, proj_out)
        else:
            self.query_proj = self.key_proj = None
        # Bilinear similarity matrix, initialised to the identity (CI paper C.1).
        self.register_parameter("W", nn.Parameter(torch.eye(self.out_dim)))

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return (h, z) for query and key.

        - h_* : encoder outputs (used by CIRO when placement == 'encoder').
        - z_* : projected outputs (the objects the contrastive loss reads).
        The momentum key encoder is detached so gradients flow to θ only.
        """
        h_q = self.query(xq)
        h_k = self.key(xk).detach()
        z_q = self.query_proj(h_q) if self.proj_enabled else h_q
        z_k = self.key_proj(h_k) if self.proj_enabled else h_k
        return {"h_q": h_q, "h_k": h_k, "z_q": z_q, "z_k": z_k}

    def bilinear(self, zq: torch.Tensor, zk: torch.Tensor) -> torch.Tensor:
        """zq^T W zk scores for the bilinear contrastive objective."""
        return zq @ self.W @ zk.T

    def update_momentum_key(self, momentum: float = 0.05) -> None:
        """xi <- m*xi + (1-m)*theta. (CIRO App. C.2, m = 0.05.)"""
        with torch.no_grad():
            for qp, kp in zip(self.query.parameters(), self.key.parameters()):
                kp.data.mul_(momentum).add_(qp.data, alpha=1.0 - momentum)

    @property
    def rep_dim(self) -> int:
        return self.query.d_rep

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["CURLModel", "curl_contrastive_loss"]