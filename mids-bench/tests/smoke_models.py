"""Forward-pass smoke test for every model in the registry.

Run on your 4090 machine (where torch + mamba-ssm are installed)::

    python tests/smoke_models.py

For each model, we:
    1. Instantiate it with its YAML defaults.
    2. Push a (B=4, L=100, F=9) random tensor through forward().
    3. Verify the output shape is (4, 4) and contains no NaN/Inf.
    4. Print param count + GPU memory if CUDA is available.

Exits non-zero if any model fails. Useful as a pre-flight check
before launching long training runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import MODEL_REGISTRY  # noqa: E402


def smoke(name: str, device: torch.device, batch_size: int = 4) -> bool:
    cfg_path = ROOT / "configs" / "model" / f"{name}.yaml"
    if not cfg_path.exists():
        print(f"  [SKIP] {name}: no config at {cfg_path}")
        return True
    cfg = yaml.safe_load(cfg_path.read_text())
    cls = MODEL_REGISTRY[cfg["name"]]
    kwargs = {k: v for k, v in cfg.items() if k != "name"}

    try:
        model = cls(**kwargs).to(device)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name}: instantiation -> {type(exc).__name__}: {exc}")
        return False

    L = kwargs.get("window_length", 100)
    F = kwargs.get("num_features", 9)
    C = kwargs.get("num_classes", 4)
    x = torch.randn(batch_size, L, F, device=device)

    model.eval()
    try:
        with torch.no_grad():
            y = model(x)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {name}: forward -> {type(exc).__name__}: {exc}")
        return False

    if y.shape != (batch_size, C):
        print(f"  [FAIL] {name}: output shape {tuple(y.shape)} != ({batch_size}, {C})")
        return False
    if torch.isnan(y).any() or torch.isinf(y).any():
        print(f"  [FAIL] {name}: output contains NaN/Inf")
        return False

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [PASS] {name:18s} | params={n_params:6.3f} M | "
          f"out={tuple(y.shape)} | range=[{y.min().item():+.3f}, {y.max().item():+.3f}]")
    return True


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Smoke-testing {len(MODEL_REGISTRY)} models on {device}")
    print("=" * 70)
    results = {name: smoke(name, device) for name in sorted(MODEL_REGISTRY)}
    print("=" * 70)
    n_fail = sum(1 for ok in results.values() if not ok)
    if n_fail:
        print(f"FAILED: {n_fail}/{len(results)}")
        return 1
    print(f"All {len(results)} models passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
