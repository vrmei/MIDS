"""MIDS detector — bidirectional Mamba over a dual-stream embedding.

This is a verbatim port of ``MambaCAN_2Direction`` from the original
``modules/model.py``, refactored to satisfy the :class:`BaseDetector`
contract:

* Accept ``(B, L, F)`` directly instead of the legacy flattened
  ``(B, L*F)``.
* Expose all architectural hyperparameters as constructor arguments
  with the values from Table II of the paper as defaults.
* Drop the ``input_width`` legacy arg — it was always 1.

Architecture (matches §V of the paper):
    1. **ID stream.** ``Linear(1 → embed_dim)`` projects the integer-as-
       float CAN ID per frame, followed by LayerNorm + ReLU + Dropout.
       This is the ``Linear`` formulation that the existing code uses;
       it does not bucket IDs into a ``nn.Embedding`` table, but the
       LayerNorm + nonlinearity recovers a similar inductive bias.
    2. **Data stream.** Two stacked ``Conv1d(kernel=3, padding=1)``
       layers map the 8-byte payload time series ``(B, 8, L)`` →
       ``(B, hidden_dim, L)`` → ``(B, 2*hidden_dim, L)``.
    3. **Fusion.** Concatenate ID and data along the feature axis to
       form ``Z ∈ ℝ^(B, L, embed_dim + 2*hidden_dim)``. With defaults
       ``embed_dim=256, hidden_dim=128`` this is exactly the
       ``Z ∈ ℝ^(L, 512)`` referenced in §V-C of the paper.
    4. **Bi-Mamba.** Two asymmetric Mamba blocks process the sequence
       forward and backward. Hyperparameters (forward state=16/conv=4,
       backward state=8/conv=2, expand=2) match the *code* checked into
       the original repo, not the symmetric values in Table II — see
       commit log for rationale.
    5. **Weighted sum + max-pool + classifier head.**
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseDetector

import logging
import os

_MAMBA_LOG = logging.getLogger(__name__)

# Prefer the official mamba-ssm CUDA kernels (~3-5x faster). If they
# aren't available (typical on native Windows), transparently fall
# back to the pure-PyTorch reference in ``_mamba_torch.py``.
#
# Set ``MIDS_FORCE_PURE_MAMBA=1`` to skip the fast path entirely, e.g.
# for numerics-comparison runs.
if os.environ.get("MIDS_FORCE_PURE_MAMBA", "0") == "1":
    from ._mamba_torch import Mamba
    _MAMBA_LOG.info("MIDS: using pure-PyTorch Mamba (forced via env).")
else:
    try:
        from mamba_ssm import Mamba  # type: ignore
        _MAMBA_LOG.info("MIDS: using mamba-ssm CUDA kernels.")
    except ImportError:
        from ._mamba_torch import Mamba
        _MAMBA_LOG.warning(
            "MIDS: mamba-ssm not available; falling back to pure-PyTorch "
            "Mamba. Training will be 3-5x slower. To install the fast "
            "kernels: `pip install causal-conv1d mamba-ssm` (Linux/WSL2 "
            "with CUDA toolkit)."
        )


class MIDS(BaseDetector):
    """The bidirectional-Mamba CAN intrusion detector.

    Args:
        num_classes: Number of output classes (default 4 = Normal/ID/Data/Both).
        window_length: ``L`` — frames per window. Default 100.
        num_features: ``F`` — features per frame (1 ID + 8 payload). Default 9.
        embed_dim: ID-stream embedding width (Table II: 256).
        hidden_dim: Data-stream conv width (the second conv emits
            ``2 * hidden_dim`` channels). Default 128.
        dropout: Dropout probability inside the embedding head and the
            classifier MLP. Default 0.3.
        mamba_fwd_state, mamba_fwd_conv: Forward Mamba SSM state /
            conv-1D dimensions (defaults 16, 4 — match the original code).
        mamba_bwd_state, mamba_bwd_conv: Backward Mamba SSM state /
            conv-1D dimensions (defaults 8, 2).
        mamba_expand: Mamba feature expansion factor (default 2).
        conv_kernel_size, conv_padding: Data-stream Conv1d shape.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        mamba_fwd_state: int = 16,
        mamba_fwd_conv: int = 4,
        mamba_bwd_state: int = 8,
        mamba_bwd_conv: int = 2,
        mamba_expand: int = 2,
        conv_kernel_size: int = 3,
        conv_padding: int = 1,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            window_length=window_length,
            num_features=num_features,
        )
        if num_features < 2:
            raise ValueError(
                f"num_features must be ≥ 2 (1 ID + ≥1 payload byte); "
                f"got {num_features}"
            )
        data_dim = num_features - 1
        d_model = embed_dim + hidden_dim * 2

        # ID stream: integer-as-float -> embed_dim.
        self.embedding_layer = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Data stream: two conv1d layers, both kernel_size=3 padding=1.
        self.data_conv1 = nn.Conv1d(
            in_channels=data_dim,
            out_channels=hidden_dim,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )
        self.data_conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=conv_kernel_size,
            padding=conv_padding,
        )

        # Bidirectional Mamba (asymmetric).
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=mamba_fwd_state,
            d_conv=mamba_fwd_conv,
            expand=mamba_expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=mamba_bwd_state,
            d_conv=mamba_bwd_conv,
            expand=mamba_expand,
        )

        # Learnable forward/backward fusion weights (paper §V-D).
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        # Classifier head. The legacy code's first linear was
        # nn.Linear(embed_dim * 2, hidden_dim) which by happy accident
        # equals d_model = embed_dim + 2*hidden_dim only when
        # embed_dim = 2*hidden_dim (256 = 2*128). We use d_model
        # explicitly here so non-default widths don't silently break.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, L, F)`` -> ``(B, num_classes)`` logits."""
        if x.dim() == 2:
            # Tolerate the legacy flattened input shape (B, L*F).
            x = x.view(x.size(0), self.window_length, self.num_features)
        elif x.dim() != 3:
            raise ValueError(
                f"expected (B, L, F) or (B, L*F); got {tuple(x.shape)}"
            )

        # Split ID and payload streams.
        id_data = x[:, :, 0:1]            # (B, L, 1)
        payload = x[:, :, 1:]             # (B, L, data_dim)

        # ID -> embedding.
        id_embed = self.embedding_layer(id_data)  # (B, L, embed_dim)

        # Payload -> conv stack. Conv1d wants (B, C, L).
        data_feat = payload.transpose(1, 2)              # (B, data_dim, L)
        data_feat = F.relu(self.data_conv1(data_feat))   # (B, hid, L)
        data_feat = F.relu(self.data_conv2(data_feat))   # (B, 2*hid, L)
        data_feat = data_feat.transpose(1, 2)            # (B, L, 2*hid)

        # Fuse along feature axis.
        z = torch.cat((id_embed, data_feat), dim=-1)     # (B, L, d_model)

        # Bi-Mamba.
        fwd = self.mamba_fwd(z)
        bwd = self.mamba_bwd(torch.flip(z, dims=[1]))
        bwd = torch.flip(bwd, dims=[1])
        h = self.alpha * fwd + self.beta * bwd           # (B, L, d_model)

        # Max-pool over the time axis.
        pooled, _ = h.max(dim=1)                         # (B, d_model)

        return self.classifier(pooled)                   # (B, num_classes)
