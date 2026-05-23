"""DCNN — In-vehicle network IDS using a deep CNN (Song et al. 2020).

Original architecture
---------------------
Song et al.'s "Inception-ResNet"-flavoured 1D CNN: stacked
``Conv1d → BN → ReLU`` blocks with residual connections, fed by a
multi-scale stem that mixes kernels of size {1, 3, 5} to capture
both local and longer-range patterns in the CAN frame stream.

Reproduction in this framework
------------------------------
We keep the multi-scale stem and the residual-style block stack but
trim depth/width to the paper-reported ~0.6 M parameter budget. Input
goes in as ``(B, F, L)`` after a single transpose adapter.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector


class _MultiScaleStem(nn.Module):
    """Concat outputs of several Conv1ds with different kernel sizes."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        per_branch = out_channels // 3
        # Distribute any remainder to the last branch so the total adds up.
        remainder = out_channels - per_branch * 3
        self.branch1 = nn.Conv1d(in_channels, per_branch, 1, padding=0)
        self.branch3 = nn.Conv1d(in_channels, per_branch, 3, padding=1)
        self.branch5 = nn.Conv1d(
            in_channels, per_branch + remainder, 5, padding=2
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        return F.relu(self.bn(out))


class _ResBlock(nn.Module):
    """Two-conv residual block with same in/out channels."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class DCNN(BaseDetector):
    """Multi-scale + residual 1D CNN for CAN intrusion detection.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        stem_channels: Output channels of the multi-scale stem.
        n_res_blocks: Number of residual blocks stacked after the stem.
        dropout: Applied before the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        stem_channels: int = 96,
        n_res_blocks: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        self.stem = _MultiScaleStem(num_features, stem_channels)
        self.blocks = nn.Sequential(
            *[_ResBlock(stem_channels) for _ in range(n_res_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(stem_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        x = x.transpose(1, 2)         # (B, F, L)
        x = self.stem(x)              # (B, stem, L)
        x = self.blocks(x)            # (B, stem, L)
        x = self.pool(x).squeeze(-1)  # (B, stem)
        x = self.dropout(x)
        return self.fc(x)
