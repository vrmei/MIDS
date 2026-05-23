"""GIDS — GAN-based Intrusion Detection System (Seo et al. 2018).

Original architecture
---------------------
Seo et al. encode CAN traffic as a 64×64 grayscale image of one-hot
ID bins, then train a GAN whose discriminator distinguishes "normal"
from "attack" image patches. Two discriminators are stacked; the
final classification draws on both.

Reproduction in this framework
------------------------------
We follow the standard fair-comparison practice (also used by the
MIDS paper, §VI-B3) of evaluating GIDS as a *supervised* classifier
that reuses only the discriminator's CNN topology with a
4-class softmax head replacing the binary real/fake head. The GAN
training loop is dropped because:

1. The unified protocol (50 epoch CE / Adam / cosine T_max=10) is the
   defining axis of comparison; a separate adversarial loop would make
   the training-protocol axis unfair.
2. The original GIDS paper itself reports the discriminator as a
   "CNN-based attack classifier" once trained, so the architectural
   capacity is what's actually being compared.

Input adapter
-------------
The legacy 64×64 image isn't directly applicable to our (B, L=100,
F=9) windows. We instead treat the window as a 1-channel 2D image of
shape ``(1, L, F)`` and run the same conv stack — the discriminator's
inductive bias (local 2D convolution + downsampling) is preserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector


class GIDS(BaseDetector):
    """Discriminator-only GIDS as a 4-class CNN classifier.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        channels: (c1, c2, c3) for the three Conv2d blocks.
        kernel_size, stride: Conv2d shape.
        dropout: Applied between conv blocks (mirrors GIDS's spectral norm
            stand-in for stable discriminator training).
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        channels: tuple = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        c1, c2, c3 = channels
        self.block1 = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        # Adapter: (B, L, F) -> (B, 1, L, F) treating the window as a
        # 1-channel 2D image with height=L and width=F.
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
