"""Resumable grid runner over (model, dataset, fold) combinations.

This is a thin wrapper over ``run_one.py``: it enumerates the requested
cross-product, *checks for already-completed runs* in
``results/runs/``, and only invokes ``run_one.py`` for the missing
ones. Because each invocation is a fresh subprocess, GPU memory is
fully released between runs.

Examples
--------
Full grid (9 models x 5 datasets x 5 folds = 225 runs)::

    python scripts/run_grid.py

Subset — just MIDS on the four public datasets::

    python scripts/run_grid.py --models mids \
        --datasets road crysys otids ctnt

Single model, all folds, one dataset::

    python scripts/run_grid.py --models cantransformer --datasets tesla

Resumability is matched by ``(model, dataset, fold)`` — the runner
looks for a directory ``results/runs/{model}_{dataset}_fold{k}_*``
that contains a ``metrics.json``. If one exists, that combo is
considered done and skipped. To force a re-run, ``rm -r`` the
relevant directory first or pass ``--force``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
RUN_ONE = ROOT / "scripts" / "run_one.py"
RESULTS_DIR = ROOT / "results" / "runs"

DEFAULT_MODELS = [
    "mids",
    "mlp",
    "cnn",
    "gids",
    "dcnn",
    "canbus_ids",
    "canshield",
    "cantransfer",
    "cantransformer",
]
DEFAULT_DATASETS = ["tesla", "road", "crysys", "otids", "ctnt"]
DEFAULT_FOLDS = [0]


def find_completed_run(model: str, dataset: str, fold: int) -> Optional[Path]:
    """Return the run directory if a metrics.json exists for this combo."""
    if not RESULTS_DIR.exists():
        return None
    pattern = f"{model}_{dataset}_fold{fold}_*"
    for d in RESULTS_DIR.glob(pattern):
        if (d / "metrics.json").exists():
            return d
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   choices=DEFAULT_MODELS, metavar="M")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   choices=DEFAULT_DATASETS, metavar="D")
    p.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS,
                   metavar="K")
    p.add_argument("--force", action="store_true",
                   help="Re-run combos that already have metrics.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned command list and exit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--continue-on-error", action="store_true",
                   help="Don't stop the grid if one run fails")
    return p.parse_args()


def build_command(
    model: str, dataset: str, fold: int, args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable, str(RUN_ONE),
        "--model", model,
        "--dataset", dataset,
        "--fold", str(fold),
        "--seed", str(args.seed),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
    ]
    return cmd


def main() -> int:
    args = parse_args()
    plan: List[Tuple[str, str, int]] = [
        (m, d, k)
        for m in args.models
        for d in args.datasets
        for k in args.folds
    ]
    total = len(plan)

    # Filter out already-completed runs.
    pending: List[Tuple[str, str, int]] = []
    skipped: List[Tuple[str, str, int]] = []
    for combo in plan:
        m, d, k = combo
        existing = find_completed_run(m, d, k)
        if existing is not None and not args.force:
            skipped.append(combo)
        else:
            pending.append(combo)

    print(f"Grid: {total} total combos | "
          f"{len(skipped)} already done | "
          f"{len(pending)} to run")
    if skipped:
        print(f"  (skip) examples: {skipped[:3]}{'...' if len(skipped) > 3 else ''}")

    if args.dry_run:
        print("\n--dry-run: would execute:")
        for m, d, k in pending:
            print("  " + " ".join(build_command(m, d, k, args)))
        return 0

    failures: List[Tuple[Tuple[str, str, int], str]] = []
    t_start_grid = time.time()
    for i, (m, d, k) in enumerate(pending, start=1):
        cmd = build_command(m, d, k, args)
        print()
        print(f"=== [{i}/{len(pending)}] {m} | {d} | fold {k} ===")
        print("  $ " + " ".join(cmd))
        t_start = time.time()
        try:
            result = subprocess.run(cmd, check=False)
            elapsed = time.time() - t_start
            if result.returncode == 0:
                print(f"  done in {elapsed:.1f}s")
            else:
                print(f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s")
                failures.append(((m, d, k), f"exit {result.returncode}"))
                if not args.continue_on_error:
                    print("  --continue-on-error not set; aborting grid.")
                    break
        except KeyboardInterrupt:
            print("  ^C — aborting grid")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION: {type(exc).__name__}: {exc}")
            failures.append(((m, d, k), repr(exc)))
            if not args.continue_on_error:
                break

    elapsed_grid = time.time() - t_start_grid
    print()
    print("=" * 60)
    print(f"Grid finished in {elapsed_grid/60:.1f} min")
    print(f"  ran      = {len(pending) - len(failures)}")
    print(f"  failed   = {len(failures)}")
    print(f"  skipped  = {len(skipped)} (had metrics.json)")
    if failures:
        print("Failures:")
        for combo, reason in failures:
            print(f"  {combo}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
