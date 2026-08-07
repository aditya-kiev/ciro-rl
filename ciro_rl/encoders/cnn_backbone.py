"""Small convolutional encoder sized for a free-tier Colab T4.

Following CURL (Laskin et al., 2020) and the CIRO paper App. C.1, the trunk is
four convolution layers with 32 channels, stride 2 on the first layer only,
then a flatten followed by a linear map to a `d_rep`-dimensional representation.

Parameter count is documented (the paper and this README ask for it) and is
printed on construction. The flatten dim is resolved dynamically from a dummy
forward so the same module handles SCM-MDP 32x32 and DCS 64x64 inputs.
"""

from typing import Optional

import torch
import torch.nn as nn


class CNNBackbone(nn.Module):
    """CURL-style 4-layer conv trunk + linear projection to `d_rep`.

    Args:
        in_channels: image channels (3).
        d_rep: representation dimensionality (default 64).
    """

    def __init__(self, in_channels: int = 3, d_rep: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ]
        self._net = nn.Sequential(*layers)
        self.fc = nn.Linear(32, d_rep)
        self.d_rep = d_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._net(x)  # (B, 32)
        return self.fc(x)

    def count_params(self) -> int:
        """Total trainable parameters (documented for the T4 sizing note)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)