"""Optional MLP projection head, mirroring the SimCLR-style head in causal-cir.

CIRO's default config uses CURL's own encoder output directly (no extra
projection head, matching the CIRO paper App. C.1). The projection head exists
for the CIRO placement ablation (Table 2: penalty on encoder output vs. on the
projection head) and to mirror the projection layers causal-cir uses.
"""

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """2-layer MLP: Linear(hid, hid?) -> BN1d -> ReLU -> Linear(hid, out).

    Args:
        in_dim: representation dimensionality feeding the head.
        hidden_dim: MLP hidden width.
        out_dim: projected dimensionality used by the contrastive loss.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # handle 1-sample batches gracefully for BatchNorm1d
        if x.dim() == 2 and x.shape[0] == 1:
            return self._forward_single(x)
        return self.net(x)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm1d needs dim>=1 with >1 sample; bypass it for n=1 (rare).
        x = self.net[0](x)
        x = self.net[3](self.net[2](x))  # ReLU(Linear)  -- skip BN
        return x