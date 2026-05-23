"""CanBus-IDS — Convolutional Adversarial Autoencoder (Hoang & Kim 2022).

Original architecture
---------------------
A semi-supervised conv-AAE: the encoder is a stack of strided Conv1d
layers, the decoder mirrors them, and an adversarial discriminator
shapes the latent prior. Detection is performed by either the
classifier head or the reconstruction-error threshold.

Reproduction in this framework
------------------------------
For the same protocol-fairness reasons as GIDS, we reproduce only the
encoder + classifier branch — the AAE training loop (with the
discriminator and the reconstruction loss) is incompatible with the
unified 50-epoch CE protocol. The encoder topology is preserved
verbatim, so the comparison still measures the architectural capacity
that defines CanBus-IDS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector


class CanBusIDS(BaseDetector):
    """Conv encoder + classifier head from Hoang & Kim 2022.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        encoder_channels: Tuple of output channels for the strided
            conv stack. Each conv halves the temporal resolution.
        latent_dim: Bottleneck width before the classifier.
        dropout: Applied before the final FC.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        encoder_channels: tuple = (32, 64, 128),
        latent_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        layers = []
        in_c = num_features
        cur_len = window_length
        for c in encoder_channels:
            layers.append(nn.Conv1d(in_c, c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm1d(c))
            layers.append(nn.ReLU(inplace=True))
            in_c = c
            cur_len = (cur_len + 1) // 2  # stride-2 with padding 1
        self.encoder = nn.Sequential(*layers)
        flat_dim = in_c * cur_len
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        x = x.transpose(1, 2)        # (B, F, L)
        x = self.encoder(x)
        z = self.bottleneck(x)
        return self.classifier(z)
