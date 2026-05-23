"""CanTransformer — Attention-based CAN intrusion detector (Jo & Kim 2024).

Original architecture
---------------------
Jo & Kim's IEEE Access 2024 model: a vanilla Transformer encoder over
the CAN frame sequence. Each frame is linearly projected to a token
embedding, summed with sinusoidal position encodings, and pushed
through ``N`` encoder layers; the sequence is mean-pooled and fed to
a softmax classifier. The paper reports the strongest non-MIDS
baseline on Tesla Tampering (F1 = 88.66%).

Reproduction in this framework
------------------------------
We use PyTorch's ``nn.TransformerEncoderLayer`` with the same
configuration the paper describes (4 heads, 4 layers, ``d_model=128``,
``feedforward=256``, GELU). Position encodings are the canonical
sinusoidal formulation from "Attention Is All You Need", computed
once at construction time and broadcast at forward time.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseDetector


class CanTransformer(BaseDetector):
    """Linear projection + sinusoidal PE + Transformer encoder + mean-pool.

    Args:
        num_classes: Output classes.
        window_length, num_features: ``L`` and ``F``.
        d_model: Token embedding width.
        nhead: Number of attention heads.
        num_layers: Number of stacked encoder layers.
        dim_feedforward: Inner FFN width inside each encoder layer.
        dropout: Encoder + classifier dropout.
        activation: ``"gelu"`` or ``"relu"`` (paper uses GELU).
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by nhead={nhead}"
            )

        self.input_proj = nn.Linear(num_features, d_model)
        self.register_buffer(
            "pos_encoding",
            self._sinusoidal_pe(window_length, d_model),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, dim: int) -> torch.Tensor:
        """Standard "Attention Is All You Need" position encoding."""
        pe = torch.zeros(seq_len, dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, L, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.window_length, self.num_features)
        x = self.input_proj(x)              # (B, L, d_model)
        x = x + self.pos_encoding[:, : x.size(1), :]
        x = self.encoder(x)                 # (B, L, d_model)
        pooled = x.mean(dim=1)              # (B, d_model)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
