"""Aggregate ``results/runs/*/metrics.json`` into LaTeX + CSV summary tables.

This produces three artefacts under ``results/tables/`` (created on demand):

    summary.csv          # raw per-(model, dataset, fold) numbers
    table_iii.tex        # paper Table III: per-model on Tesla, mean +/- std
    table_v.tex          # paper Table V: per-dataset for the chosen model

Each numeric cell shows ``mean +- std`` over the available folds. Missing
combinations are reported as ``--``. The metric set matches what
``run_one.py`` writes:

    Precision, Recall, F1, Accuracy, FPR, FNR, AUC, latency_ms, params, flops

Usage::

    python scripts/make_table.py
    python scripts/make_table.py --table-v-model mids
    python scripts/make_table.py --results-dir results/runs --out-dir results/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# Column order shared across both LaTeX tables.
METRIC_COLS = ["precision", "recall", "f1", "accuracy", "fpr", "fnr", "auc"]
COST_COLS = ["latency_ms", "params", "flops"]
PCT_METRICS = {"precision", "recall", "f1", "accuracy", "fpr", "fnr", "auc"}

# Pretty model labels (matches paper Table III).
MODEL_DISPLAY = {
    "gids": "GIDS",
    "canshield": "CANShield",
    "canbus_ids": "CanBus-IDS",
    "dcnn": "DCNN",
    "cantransfer": "CANTransfer",
    "cantransformer": "CanTransformer",
    "mlp": "MLP",
    "cnn": "CNN",
    "mids": "MIDS (Ours)",
}
DATASET_DISPLAY = {
    "tesla": "Tesla (Ours)",
    "road": "ROAD",
    "crysys": "CrySyS",
    "otids": "OTIDS",
    "ctnt": "CT\\&T",
}

# Paper Table III ordering: SOTA models first, foundational, then ours.
TABLE_III_MODEL_ORDER = [
    "gids", "canshield", "canbus_ids", "dcnn",
    "cantransfer", "cantransformer",
    "mlp", "cnn",
    "mids",
]
# Paper Table V ordering.
TABLE_V_DATASET_ORDER = ["tesla", "road", "crysys", "otids", "ctnt"]


def collect_runs(results_dir: Path) -> List[dict]:
    """Walk results/runs/* and collect per-run metric+config dicts."""
    rows: List[dict] = []
    if not results_dir.exists():
        return rows
    for run_dir in sorted(results_dir.iterdir()):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        args = config.get("args", {})
        row = {
            "run_id": run_dir.name,
            "model": args.get("model"),
            "dataset": args.get("dataset"),
            "fold": args.get("fold"),
        }
        # Fall back to parsing the run_id when config.json is missing.
        if row["model"] is None or row["dataset"] is None:
            parsed = _parse_run_id(run_dir.name)
            row.update(parsed)
        for k in METRIC_COLS + COST_COLS:
            row[k] = metrics.get(k)
        rows.append(row)
    return rows


def _parse_run_id(run_id: str) -> Dict[str, Optional[object]]:
    """Best-effort parse of '{model}_{dataset}_fold{k}_{ts}'."""
    parts = run_id.split("_")
    out: Dict[str, Optional[object]] = {"model": None, "dataset": None, "fold": None}
    for i, p in enumerate(parts):
        if p.startswith("fold") and p[4:].isdigit():
            out["fold"] = int(p[4:])
            if i >= 2:
                out["model"] = "_".join(parts[:i - 1])
                out["dataset"] = parts[i - 1]
            break
    return out


def write_summary_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "model", "dataset", "fold"] + METRIC_COLS + COST_COLS
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def aggregate(
    rows: List[dict], by: Tuple[str, ...] = ("model", "dataset"),
) -> Dict[tuple, Dict[str, Tuple[float, float, int]]]:
    """Group rows by ``by`` and compute per-metric (mean, std, n) tuples."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(b) for b in by)
        if any(v is None for v in key):
            continue
        groups[key].append(r)

    out: Dict[tuple, Dict[str, Tuple[float, float, int]]] = {}
    for key, grp in groups.items():
        agg: Dict[str, Tuple[float, float, int]] = {}
        for col in METRIC_COLS + COST_COLS:
            vals = [r[col] for r in grp
                    if r.get(col) is not None and not _is_nan(r.get(col))]
            n = len(vals)
            if n == 0:
                agg[col] = (math.nan, math.nan, 0)
            else:
                m = sum(vals) / n
                v = sum((x - m) ** 2 for x in vals) / max(n - 1, 1) if n > 1 else 0.0
                agg[col] = (m, math.sqrt(v), n)
        out[key] = agg
    return out


def _is_nan(x: object) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _fmt_pct(mean: float, std: float, n: int) -> str:
    if n == 0 or math.isnan(mean):
        return "--"
    if n == 1:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} $\\pm$ {std * 100:.2f}"


def _fmt_num(mean: float, std: float, n: int, fmt: str) -> str:
    if n == 0 or math.isnan(mean):
        return "--"
    if n == 1:
        return f"{mean:{fmt}}"
    return f"{mean:{fmt}} $\\pm$ {std:{fmt}}"


def _fmt_cell(metric: str, mean: float, std: float, n: int) -> str:
    if metric in PCT_METRICS:
        return _fmt_pct(mean, std, n)
    if metric == "latency_ms":
        return _fmt_num(mean, std, n, ".3f")
    if metric == "flops":
        return _fmt_num(mean, std, n, ".4f")
    return _fmt_num(mean, std, n, ".3f")


def write_table_iii(
    agg: Dict[tuple, Dict[str, Tuple[float, float, int]]],
    out_path: Path,
    target_dataset: str = "tesla",
) -> None:
    """Per-model summary on a single dataset (default Tesla)."""
    cols = ["precision", "recall", "f1", "accuracy", "fpr", "fnr", "auc"]
    headers = ["Model"] + [c.upper() for c in cols]

    lines: List[str] = []
    lines.append("\\begin{tabular}{l" + "r" * len(cols) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for m in TABLE_III_MODEL_ORDER:
        cells = [MODEL_DISPLAY.get(m, m)]
        a = agg.get((m, target_dataset))
        for col in cols:
            if a is None:
                cells.append("--")
            else:
                mean, std, n = a[col]
                cells.append(_fmt_cell(col, mean, std, n))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table_v(
    agg: Dict[tuple, Dict[str, Tuple[float, float, int]]],
    out_path: Path,
    target_model: str = "mids",
) -> None:
    """Per-dataset summary for a single model (default MIDS)."""
    cols = ["precision", "recall", "f1", "accuracy", "fpr", "fnr"]
    headers = ["Dataset"] + [c.upper() for c in cols]

    lines: List[str] = []
    lines.append("\\begin{tabular}{l" + "r" * len(cols) + "}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")
    for d in TABLE_V_DATASET_ORDER:
        cells = [DATASET_DISPLAY.get(d, d)]
        a = agg.get((target_model, d))
        for col in cols:
            if a is None:
                cells.append("--")
            else:
                mean, std, n = a[col]
                cells.append(_fmt_cell(col, mean, std, n))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    parser.add_argument("--table-iii-dataset", default="tesla",
                        help="Dataset for the per-model table (default tesla)")
    parser.add_argument("--table-v-model", default="mids",
                        help="Model for the per-dataset table (default mids)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    rows = collect_runs(results_dir)
    if not rows:
        print(f"No runs found under {results_dir}.")
        return 0
    print(f"Collected {len(rows)} run(s) from {results_dir}")

    summary_path = out_dir / "summary.csv"
    write_summary_csv(rows, summary_path)
    print(f"  wrote {summary_path}")

    agg = aggregate(rows, by=("model", "dataset"))

    table_iii_path = out_dir / "table_iii.tex"
    write_table_iii(agg, table_iii_path, target_dataset=args.table_iii_dataset)
    print(f"  wrote {table_iii_path} (per-model on {args.table_iii_dataset})")

    table_v_path = out_dir / "table_v.tex"
    write_table_v(agg, table_v_path, target_model=args.table_v_model)
    print(f"  wrote {table_v_path} (per-dataset for {args.table_v_model})")

    print()
    print("Summary by (model, dataset):")
    for (m, d), a in sorted(agg.items()):
        f1 = a["f1"]
        print(f"  {m:18s} | {d:8s} | F1 {_fmt_pct(*f1):>22s} (n={f1[2]})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
