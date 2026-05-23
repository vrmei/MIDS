"""Foundational MLP baseline.

Per §VI-B2 of the paper, the MLP is included as a "foundational" sanity
check — a featureless three-layer network applied to the flattened
window. It is expected to do poorly on this task (the paper reports
F1 = 38.48% on the Tesla dataset) and serves mainly to bound the
information content of windowed CAN data without sequence-aware
inductive biases.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDetector


class MLP(BaseDetector):
    """3-layer fully-connected classifier on the flattened window.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``; the input is
            flattened to ``L * F`` before the first FC.
        hidden_dims: Tuple of (h1, h2) for the two hidden layers.
        dropout: Applied between FC layers.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        hidden_dims: tuple = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        h1, h2 = hidden_dims
        in_dim = window_length * num_features
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        return self.net(x)
