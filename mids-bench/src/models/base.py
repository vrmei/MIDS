"""Abstract base class for all detection models in the framework."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseDetector(nn.Module, ABC):
    """Common interface every detector must satisfy.

    Subclasses receive ``(B, L, F)`` tensors and emit ``(B, num_classes)``
    logits. ``num_classes``, ``window_length``, and ``num_features`` are
    stored on the instance so the trainer and metric code can introspect
    a model without inspecting forward signatures.
    """

    def __init__(
        self,
        num_classes: int = 4,
        window_length: int = 100,
        num_features: int = 9,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.window_length = window_length
        self.num_features = num_features

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return ``(B, num_classes)`` logits for input ``(B, L, F)``."""
        raise NotImplementedError
