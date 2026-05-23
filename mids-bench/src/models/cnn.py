"""Foundational 1D-CNN baseline.

Per §VI-B2, CNN is the second foundational sanity model. The window
is treated as a multivariate time series with ``F`` channels and
``L`` time steps, and a small Conv1d stack feeds a global-pool +
classifier head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector


class CNN(BaseDetector):
    """Conv1d → Conv1d → AdaptiveMaxPool → FC.

    Channels = ``num_features``; time = ``window_length``.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        channels: (c1, c2) channel widths for the two conv layers.
        kernel_size, padding: Same for both convs.
        dropout: Applied before the final FC.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        channels: tuple = (64, 128),
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        c1, c2 = channels
        self.conv1 = nn.Conv1d(num_features, c1, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size, padding=padding)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        # (B, L, F) -> (B, F, L) for Conv1d.
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
