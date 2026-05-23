"""Cross-entropy loss with dynamic class weighting.

Per §VI-B4 of the paper, weights are recomputed *per fold* from the
training-set label distribution. We use the scikit-learn
``"balanced"`` formula:

    w_c = N / (C * n_c)

where ``N`` is the total number of training windows, ``C`` is the
number of classes, and ``n_c`` is the count of class ``c``. With this
formula, classes that are present have a weight inversely proportional
to their frequency, and the average weight equals 1.

Classes that don't appear at all (``n_c = 0``) get weight 1.0 — they
contribute nothing to the loss anyway because no example is labelled
that class, but we avoid division-by-zero.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn


def compute_balanced_weights(
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Sklearn-style balanced class weights.

    Args:
        labels: 1-D array of integer class labels.
        num_classes: Number of distinct class IDs (some may be absent).

    Returns:
        Array of length ``num_classes`` with weight per class.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    n_total = labels.shape[0]
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = n_total / (num_classes * counts[present])
    return weights


def build_weighted_ce(
    labels: np.ndarray,
    num_classes: int,
    device: Optional[torch.device] = None,
) -> nn.CrossEntropyLoss:
    """Construct a :class:`nn.CrossEntropyLoss` with balanced weights.

    Args:
        labels: 1-D array of training-set labels.
        num_classes: Number of classes.
        device: Where to place the weight tensor.

    Returns:
        A configured loss module ready to take ``(logits, target)``.
    """
    weights = compute_balanced_weights(labels, num_classes)
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        weight_tensor = weight_tensor.to(device)
    return nn.CrossEntropyLoss(weight=weight_tensor)
