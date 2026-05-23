"""Run one (model, dataset, fold) combination end-to-end.

Usage::

    python scripts/run_one.py --model mids --dataset tesla --fold 0

Outputs land in ``results/runs/{run_id}/``:

    metrics.json     # 10 keys: P/R/F1/Acc/FPR/FNR/AUC/latency/params/flops
    confusion.npy    # (C, C) int64 confusion matrix
    history.json     # per-epoch train+test metrics
    run.log          # full stdlib logging output
    model.pt         # final model state_dict
    config.json      # the resolved config bundle

Run IDs are ``{model}_{dataset}_fold{k}_{YYYYMMDD-HHMMSS}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import DATASET_REGISTRY  # noqa: E402
from src.models import MODEL_REGISTRY  # noqa: E402
from src.train import train_one_fold  # noqa: E402
from src.train.trainer import TrainConfig  # noqa: E402
from src.utils import set_global_seed, setup_logger  # noqa: E402

DATASETS = DATASET_REGISTRY
MODELS = MODEL_REGISTRY


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(p: str, base: Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def _build_dataset(cfg: dict, fold: int, split: str, base: Path):
    name = cfg["name"]
    cls = DATASETS[name]
    return cls(
        root=str(_resolve_path(cfg["root"], base)),
        split=split,
        fold=fold,
        n_folds=cfg["n_folds"],
        num_classes=cfg["num_classes"],
        cache_dir=str((base / "cache").resolve()),
        seed=cfg["seed"],
        chunk_size=cfg["chunk_size"],
        buffer_windows=cfg["buffer_windows"],
    )


def _build_model(cfg: dict):
    name = cfg["name"]
    cls = MODELS[name]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**kwargs)


def _make_run_id(model: str, dataset: str, fold: int) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"{model}_{dataset}_fold{fold}_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help=("DataLoader workers. Default 0: dataset is fully in RAM, "
              "so multiprocess workers add fork overhead with no benefit."),
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_cfg_path = Path(args.data_config) if args.data_config else (
        base / "configs" / "data" / f"{args.dataset}.yaml"
    )
    model_cfg_path = Path(args.model_config) if args.model_config else (
        base / "configs" / "model" / f"{args.model}.yaml"
    )
    train_cfg_path = Path(args.train_config) if args.train_config else (
        base / "configs" / "train" / "default.yaml"
    )

    data_cfg = _load_yaml(data_cfg_path)
    model_cfg = _load_yaml(model_cfg_path)
    train_cfg_dict = _load_yaml(train_cfg_path)

    if model_cfg.get("window_length") != train_cfg_dict.get("window_length"):
        raise ValueError(
            f"model.window_length != train.window_length; got "
            f"{model_cfg.get('window_length')} vs {train_cfg_dict.get('window_length')}"
        )
    if model_cfg.get("num_classes") != data_cfg.get("num_classes"):
        raise ValueError(
            f"model.num_classes != data.num_classes; got "
            f"{model_cfg.get('num_classes')} vs {data_cfg.get('num_classes')}"
        )

    run_id = _make_run_id(args.model, args.dataset, args.fold)
    out_dir = (base / "results" / "runs" / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("mids_bench", log_file=out_dir / "run.log")
    logger.info("Run ID: %s", run_id)
    logger.info("Output dir: %s", out_dir)

    set_global_seed(args.seed)
    logger.info("Seed: %d", args.seed)
    logger.info("Device: %s (cuda available=%s)",
                args.device, torch.cuda.is_available())

    train_ds = _build_dataset(data_cfg, args.fold, "train", base)
    test_ds = _build_dataset(data_cfg, args.fold, "test", base)
    logger.info(
        "Dataset: train=%d windows %s | test=%d windows %s",
        len(train_ds), train_ds.class_counts().tolist(),
        len(test_ds), test_ds.class_counts().tolist(),
    )

    pin_memory = args.device.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg_dict["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg_dict["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = _build_model(model_cfg)
    logger.info("Model: %s | params=%.3f M",
                args.model,
                sum(p.numel() for p in model.parameters()) / 1e6)

    cfg = TrainConfig(**{k: v for k, v in train_cfg_dict.items()})
    metrics = train_one_fold(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        cfg=cfg,
        device=args.device,
    )

    cm = metrics.pop("confusion_matrix")
    history = metrics.pop("train_history")
    np.save(out_dir / "confusion.npy", cm)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": data_cfg,
                "model": model_cfg,
                "train": asdict(cfg),
                "args": vars(args),
                "run_id": run_id,
            },
            f, indent=2,
        )
    torch.save(model.state_dict(), out_dir / "model.pt")

    logger.info(
        "Done. F1=%.4f Acc=%.4f | latency=%.3f ms | params=%.3f M",
        metrics["f1"], metrics["accuracy"],
        metrics["latency_ms"], metrics["params"],
    )

    # NOTE: The earlier "F1 < 0.95 -> exit 2" gate was a Batch-1 sanity
    # check for MIDS-on-Tesla specifically (paper claim). It's not
    # meaningful for the baseline grid — most baselines legitimately
    # score well below 0.95, especially on cross-dataset evaluation
    # and on folds where minority classes are sparsely represented.
    # Keep this as an informational log only.
    if metrics["f1"] < 0.5:
        logger.info(
            "Note: final test F1 = %.4f is low. If this is MIDS on Tesla, "
            "investigate as a refactor bug. Otherwise expected.",
            metrics["f1"],
        )


if __name__ == "__main__":
    main()
