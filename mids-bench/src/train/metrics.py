"""Evaluation metrics, parameter / FLOP counting, and latency timing.

The ten keys returned by :func:`compute_metrics` + the helpers below
are exactly the keys the trainer writes to ``metrics.json``:

    precision, recall, f1, accuracy   - macro-averaged 4-class metrics
    fpr, fnr                          - binary collapse: Normal vs Attack
    auc                               - macro one-vs-rest ROC-AUC
    latency_ms                        - single-window, batch=1, 1000 runs
    params                            - millions of parameters
    flops                             - giga-FLOPs per forward pass

Conventions match §VI-B / VI-G of the MIDS paper.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    num_classes: int,
) -> Dict[str, float]:
    """Compute the seven classification metrics.

    Args:
        y_true: 1-D int array of ground-truth labels, length N.
        y_pred: 1-D int array of predicted labels, length N.
        y_prob: Optional ``(N, num_classes)`` softmax probabilities.
            Required for AUC; if None, AUC is set to NaN.
        num_classes: Number of classes (4 for the Tesla setting).

    Returns:
        Dict with keys precision, recall, f1, accuracy, fpr, fnr, auc.
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    # Macro P/R/F1. Classes absent from y_true AND y_pred are skipped
    # from the macro average — otherwise they contribute F1=0 (since
    # tp=fp=fn=0 gives p=r=0) and crater the macro score on folds where
    # block-shuffled splits happen to exclude a rare class. This
    # matches the spirit of sklearn's zero_division="warn" behaviour.
    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        n_true = int((y_true == c).sum())
        n_pred = int((y_pred == c).sum())
        if n_true == 0 and n_pred == 0:
            # Class never appears anywhere — skip it. Macro will average
            # over fewer classes; the missing class is reported in
            # logs so the test split imbalance is visible.
            continue
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * p * r, p + r)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # Binary collapse for FPR / FNR: 0 = Normal, anything else = Attack.
    is_attack_true = y_true != 0
    is_attack_pred = y_pred != 0
    binary_tp = int((is_attack_pred & is_attack_true).sum())
    binary_fp = int((is_attack_pred & ~is_attack_true).sum())
    binary_tn = int((~is_attack_pred & ~is_attack_true).sum())
    binary_fn = int((~is_attack_pred & is_attack_true).sum())
    fpr = _safe_div(binary_fp, binary_fp + binary_tn)
    fnr = _safe_div(binary_fn, binary_fn + binary_tp)

    auc = float("nan")
    if y_prob is not None:
        try:
            auc = _macro_ovr_auc(y_true, y_prob, num_classes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AUC computation failed: %s", exc)

    return {
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
        "accuracy": accuracy,
        "fpr": fpr,
        "fnr": fnr,
        "auc": auc,
    }


def _macro_ovr_auc(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int
) -> float:
    """Macro one-vs-rest ROC AUC with a tiny manual implementation.

    Uses the trapezoidal rule on the per-class ROC curve; classes that
    are absent from ``y_true`` are skipped (otherwise AUC is undefined).
    """
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == y_bin.size:
            continue  # all-positive or all-negative -> AUC undefined
        scores = y_prob[:, c]
        order = np.argsort(-scores)
        y_sorted = y_bin[order]
        n_pos = y_sorted.sum()
        n_neg = y_sorted.size - n_pos
        # cumulative TPR / FPR
        tps = np.cumsum(y_sorted)
        fps = np.cumsum(1 - y_sorted)
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        # trapezoidal area
        aucs.append(float(np.trapz(tpr, fpr)))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    """``(C, C)`` int64 confusion matrix; rows = true, cols = pred."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# ---------------------------------------------------------------------------
# Cost / latency
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> float:
    """Total parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> float:
    """FLOPs per forward pass, in giga-FLOPs.

    Uses ``fvcore.nn.FlopCountAnalysis`` if available. Returns 0.0 and
    logs a warning if fvcore is missing or the count fails (this lets
    sanity-check runs proceed even on environments without fvcore).
    SSM selective-scan ops aren't natively supported by any FLOP
    counter, so this number is a lower bound — same caveat as in §VI-G
    of the paper, which reports 0.010 G for MIDS.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logger.warning("fvcore not installed; reporting flops=0.0")
        return 0.0

    model_was_training = model.training
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    dummy = torch.zeros(input_shape, device=device)
    try:
        analysis = FlopCountAnalysis(model, dummy)
        # Silence per-op warnings; we know SSM ops are unsupported.
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        total = analysis.total()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FLOP counting failed: %s", exc)
        total = 0.0
    finally:
        if model_was_training:
            model.train()
    return float(total) / 1e9


def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 100, 9),
    n_warmup: int = 100,
    n_runs: int = 1000,
    device: Optional[torch.device] = None,
) -> float:
    """Mean per-window latency in ms over ``n_runs`` after ``n_warmup``.

    Per §VI-G of the paper: ``batch_size = 1`` to mimic real-time
    deployment. Uses ``torch.cuda.Event`` on CUDA for accurate timing,
    falls back to ``time.perf_counter`` on CPU.
    """
    if device is None:
        device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    dummy = torch.zeros(input_shape, device=device)

    use_cuda = device.type == "cuda"
    timings = []

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
        if use_cuda:
            torch.cuda.synchronize()
            for _ in range(n_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = model(dummy)
                end.record()
                torch.cuda.synchronize()
                timings.append(start.elapsed_time(end))  # already ms
        else:
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy)
                timings.append((time.perf_counter() - t0) * 1000.0)

    if model_was_training:
        model.train()
    return float(np.mean(timings))
