"""5-fold trainer for the MIDS benchmark framework.

This module implements :func:`train_one_fold`, which is the single
entry point the orchestrator uses. It owns:

* The fixed unified-protocol hyperparameters from §VI-B3 of the paper:
  50 epochs, Adam, lr=1e-4, cosine annealing T_max=10, gradient clip
  ℓ2-norm 1.0, batch size 1024.
* The dynamic class-weighting recomputed from the training set.
* Per-epoch logging of train loss / accuracy and test macro metrics.
* Final-epoch evaluation that produces the metrics dict + confusion
  matrix the orchestrator persists to disk.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import build_weighted_ce
from .metrics import (
    compute_metrics,
    confusion_matrix,
    count_flops,
    count_params,
    measure_latency,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Frozen hyperparameter bundle (§VI-B3)."""

    n_epochs: int = 50
    lr: float = 1e-4
    cosine_t_max: int = 10
    grad_clip_norm: float = 1.0
    batch_size: int = 1024
    num_classes: int = 4
    window_length: int = 100
    num_features: int = 9
    use_balanced_weights: bool = True
    latency_warmup: int = 100
    latency_runs: int = 1000
    eval_every: int = 1
    progress_bar: bool = True
    extra: Dict = field(default_factory=dict)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    progress: bool = True,
    desc: str = "eval",
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over ``loader`` and compute the seven cls metrics."""
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    iterator = loader
    if progress:
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch")
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes)
    return metrics, y_true, y_pred, y_prob


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    device: str = "cuda",
) -> Dict[str, object]:
    """Train one model on one fold and return its metric bundle.

    The returned dict is the union of the seven classification metrics
    (precision, recall, f1, accuracy, fpr, fnr, auc) and three cost
    metrics (latency_ms, params, flops). It also contains
    ``confusion_matrix`` (C x C int64 array) and ``train_history`` (list
    of per-epoch dicts) for downstream inspection.
    """
    device_t = torch.device(device)
    model = model.to(device_t)

    # Loss with balanced weights (computed from THIS fold's train set).
    if cfg.use_balanced_weights:
        ds = train_loader.dataset
        if hasattr(ds, "_labels"):
            train_labels = ds._labels.cpu().numpy()
        else:
            train_labels = np.concatenate(
                [batch[1].numpy() for batch in train_loader]
            )
        criterion = build_weighted_ce(train_labels, cfg.num_classes, device_t)
        logger.info(
            "Class weights (balanced): %s",
            criterion.weight.detach().cpu().numpy().tolist(),
        )
    else:
        criterion = nn.CrossEntropyLoss().to(device_t)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.cosine_t_max
    )

    history = []
    final_metrics: Dict[str, float] = {}
    final_y_true = final_y_pred = None

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_start = time.time()

        train_iter = train_loader
        if cfg.progress_bar:
            train_iter = tqdm(
                train_loader,
                desc=f"epoch {epoch}/{cfg.n_epochs} train",
                leave=False,
                unit="batch",
            )

        for x, y in train_iter:
            x = x.to(device_t, non_blocking=True)
            y = y.to(device_t, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_norm
            )
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=-1)
            epoch_correct += (preds == y).sum().item()
            epoch_total += y.size(0)

            if cfg.progress_bar:
                train_iter.set_postfix(
                    loss=f"{epoch_loss / max(epoch_total, 1):.4f}",
                    acc=f"{epoch_correct / max(epoch_total, 1):.4f}",
                )

        scheduler.step()

        train_loss = epoch_loss / max(epoch_total, 1)
        train_acc = epoch_correct / max(epoch_total, 1)
        epoch_time = time.time() - t_start

        is_last = epoch == cfg.n_epochs
        do_eval = is_last or (epoch % max(cfg.eval_every, 1) == 0)

        if do_eval:
            eval_metrics, y_true, y_pred, _ = _evaluate(
                model, test_loader, device_t, cfg.num_classes,
                progress=cfg.progress_bar,
                desc=f"epoch {epoch}/{cfg.n_epochs} eval",
            )
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | "
                "test P=%.4f R=%.4f F1=%.4f Acc=%.4f | %.1fs",
                epoch, cfg.n_epochs, train_loss, train_acc,
                eval_metrics["precision"], eval_metrics["recall"],
                eval_metrics["f1"], eval_metrics["accuracy"],
                epoch_time,
            )
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test": eval_metrics,
                "epoch_time_s": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            })
            final_metrics = eval_metrics
            final_y_true, final_y_pred = y_true, y_pred
        else:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | (eval skipped) | %.1fs",
                epoch, cfg.n_epochs, train_loss, train_acc, epoch_time,
            )
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test": None,
                "epoch_time_s": epoch_time,
                "lr": optimizer.param_groups[0]["lr"],
            })

    # Cost metrics (params / flops / latency) -- one-shot at the end.
    params_m = count_params(model)
    flops_g = count_flops(
        model,
        input_shape=(1, cfg.window_length, cfg.num_features),
        device=device_t,
    )
    latency_ms = measure_latency(
        model,
        input_shape=(1, cfg.window_length, cfg.num_features),
        n_warmup=cfg.latency_warmup,
        n_runs=cfg.latency_runs,
        device=device_t,
    )
    logger.info(
        "Cost: params=%.3f M | flops=%.4f G | latency=%.3f ms",
        params_m, flops_g, latency_ms,
    )

    cm = confusion_matrix(final_y_true, final_y_pred, cfg.num_classes)

    out: Dict[str, object] = dict(final_metrics)
    out["latency_ms"] = latency_ms
    out["params"] = params_m
    out["flops"] = flops_g
    out["confusion_matrix"] = cm
    out["train_history"] = history
    return out
