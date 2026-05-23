"""CANShield — Signal-level CNN + LSTM detector (Shahriar et al. 2023).

Original architecture
---------------------
CANShield runs *parallel signal streams*: each periodic CAN signal
(grouped by ID) is processed by its own per-stream CNN, and the
per-stream CNN outputs are aggregated by a shared LSTM that models
cross-signal temporal dependencies.

Reproduction in this framework
------------------------------
The original 38-stream layout depends on a per-vehicle CAN-DBC that
groups raw bytes into named signals — information that is not
available for arbitrary CAN traces in the unified-protocol setting.
We collapse the multi-stream stage into a *single* signal-level CNN
that operates over all 9 features simultaneously, while keeping the
defining LSTM aggregator that gives CANShield its sequential
inductive bias. The convolution depth and LSTM hidden size are
chosen to roughly match the per-window parameter count reported in
the original paper (~0.2 M).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector


class CANShield(BaseDetector):
    """Conv1d feature extractor → LSTM → classifier.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        conv_channels: (c1, c2) for the two-layer signal CNN.
        lstm_hidden: Hidden size of the LSTM aggregator.
        lstm_layers: Number of stacked LSTM layers.
        bidirectional: Whether the LSTM is bidirectional.
        dropout: Applied between conv blocks and on top of the LSTM.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        conv_channels: tuple = (32, 64),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        c1, c2 = conv_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, c1, kernel_size=3, padding=1),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        x = x.transpose(1, 2)        # (B, F, L)
        feats = self.cnn(x)          # (B, c2, L)
        feats = feats.transpose(1, 2)  # (B, L, c2)
        out, _ = self.lstm(feats)    # (B, L, H or 2H)
        last = out[:, -1, :]         # take the last timestep
        last = self.dropout(last)
        return self.fc(last)
