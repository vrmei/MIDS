"""CANTransfer — Conv-LSTM with one-shot transfer learning (Tariq 2020).

Original architecture
---------------------
Tariq et al. propose a stacked ConvLSTM2D classifier reshaping the
window into a 4-D ``(B, T, C, H, W)`` tensor for spatial-temporal
modelling, then fine-tune on novel attack types using one-shot
learning (a single labelled sample per new class).

Reproduction in this framework
------------------------------
Two simplifications, both flagged as out-of-scope-for-Batch-2 in the
spec:

1. **No one-shot transfer.** The transfer-learning loop requires
   per-attack-type meta-batches that don't fit the unified 50-epoch
   protocol. We train CANTransfer end-to-end on the labelled training
   fold like every other baseline; the architectural backbone is what
   gets compared.

2. **Conv1d + LSTM instead of ConvLSTM2D.** PyTorch lacks a stable
   ConvLSTM2D layer, and rolling our own is its own can of worms. We
   use the same factorisation everyone else uses in CAN-IDS papers:
   a Conv1d local-feature extractor followed by a stacked LSTM
   temporal model. This is the same factorisation behind CANShield;
   the difference is that CANTransfer uses a *deeper* LSTM stack
   (2 layers, default) and a wider FC head, mirroring the higher
   capacity reported in the original paper.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import BaseDetector


class CANTransfer(BaseDetector):
    """Conv1d front-end + 2-layer LSTM + 2-layer FC head.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        conv_channels: (c1, c2) for the front-end.
        lstm_hidden: LSTM hidden width.
        lstm_layers: Stacked LSTM depth (default 2 — distinguishes
            CANTransfer from CANShield's 1-layer LSTM).
        head_hidden: Width of the FC head's hidden layer.
        dropout: Applied to LSTM (between layers) and the FC head.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        conv_channels: tuple = (32, 64),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        c1, c2 = conv_channels
        self.frontend = nn.Sequential(
            nn.Conv1d(num_features, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        x = x.transpose(1, 2)              # (B, F, L)
        x = self.frontend(x)               # (B, c2, L)
        x = x.transpose(1, 2)              # (B, L, c2)
        out, _ = self.lstm(x)              # (B, L, H)
        last = out[:, -1, :]
        return self.head(last)
