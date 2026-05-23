# mids-bench

Unified benchmark framework for CAN-bus intrusion detection: 9 detector
architectures across 5 datasets under one training protocol, with
resumable grid execution and LaTeX-ready aggregation.

## Layout

```
mids-bench/
├── configs/
│   ├── data/         # one YAML per dataset (root path, fold count, ...)
│   ├── model/        # one YAML per detector (architecture hyperparams)
│   └── train/        # default.yaml — frozen unified protocol
├── src/
│   ├── datasets/     # CANDataset base + 5 dataset parsers
│   ├── models/       # BaseDetector + 9 detectors (MIDS + 8 baselines)
│   ├── train/        # trainer / metrics / losses
│   └── utils/        # seed / logger / block-shuffle splits
├── scripts/
│   ├── run_one.py            # one (model, dataset, fold) combo
│   ├── run_grid.py           # cross-product, resumable
│   ├── make_table.py         # LaTeX + CSV aggregation
│   ├── setup_windows.ps1     # one-shot Windows env install
│   └── download_datasets.sh  # WSL/git-bash helper
├── tests/
│   └── smoke_models.py       # forward-pass sanity for every model
├── cache/                    # gitignored: parsed windows + fold indices
├── results/runs/             # gitignored: per-run metrics.json + ckpt
└── results/tables/           # LaTeX + CSV aggregates
```

## Models

| name | file | source |
|---|---|---|
| `mids` | `src/models/mids.py` | Liu et al. TIFS (this paper) |
| `mlp` / `cnn` | foundational | — |
| `gids` | `src/models/gids.py` | Seo et al. PST 2018 |
| `dcnn` | `src/models/dcnn.py` | Song et al. Veh. Commun. 2020 |
| `canbus_ids` | `src/models/canbus_ids.py` | Hoang & Kim 2022 |
| `canshield` | `src/models/canshield.py` | Shahriar et al. IoT-J 2023 |
| `cantransfer` | `src/models/cantransfer.py` | Tariq et al. SAC 2020 |
| `cantransformer` | `src/models/cantransformer.py` | Jo & Kim IEEE Access 2024 |

All models satisfy `BaseDetector(input=(B, L=100, F=9), output=(B, 4))`.
Add a new one by dropping a file under `src/models/` and registering it
in `src/models/__init__.py::MODEL_REGISTRY`.

## Datasets

| name | file | data layout |
|---|---|---|
| `tesla` | `src/datasets/tesla.py` | flat `.npy` (N, L*F+1), zero-copy mmap |
| `road` | `src/datasets/road.py` | candump `.log` + `capture_metadata.json` |
| `crysys` | `src/datasets/crysys.py` | per-recording CSV, label column |
| `otids` | `src/datasets/otids.py` | 4 `.txt` files (one per attack class) |
| `ctnt` | `src/datasets/ctnt.py` | per-vehicle CSV folders |

Window-labelling strategy is per-dataset (`WINDOW_LABEL_STRATEGY` class
attribute). Tesla / OTIDS / CT&T use `"majority"`; ROAD / CrySyS use
`"any-attack"` because their attacks are sparse within a window.

## Quick start (Windows + RTX 40-series)

```powershell
# 1. Install env. Creates conda env 'mids-bench' with PyTorch CUDA 12.1.
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
conda activate mids-bench

# 2. Smoke-test all models.
python tests\smoke_models.py

# 3. Run one combination.
python scripts\run_one.py --model mlp --dataset otids --fold 0

# 4. Run a grid (resumable: re-running picks up where you left off).
python scripts\run_grid.py --models mlp cnn gids dcnn canbus_ids ^
    canshield cantransfer cantransformer --datasets otids road

# 5. Aggregate results into LaTeX.
python scripts\make_table.py
# -> results\tables\{summary.csv, table_iii.tex, table_v.tex}
```

## Quick start (Linux + CUDA)

For the optional fast Mamba kernels (mamba-ssm + causal-conv1d), Linux
with a working CUDA toolchain is the path of least resistance:

```bash
pip install -r requirements.txt   # full stack including mamba-ssm
python scripts/run_one.py --model mids --dataset tesla --fold 0
```

On Windows, `requirements-minimal.txt` skips mamba-ssm and the codebase
auto-falls-back to a pure-PyTorch Mamba (~3-5x slower but works).

## Unified training protocol (§VI-B3 of the paper)

Frozen across all baselines and datasets — the comparison axis IS the
architecture:

- 50 epochs, Adam, lr = 1e-4
- Cosine annealing LR, T_max = 10
- Gradient clip ℓ2-norm = 1.0
- Batch size 1024
- Dynamic class weighting (sklearn balanced) recomputed per fold
- Block-shuffled 5-fold CV with 1000-window chunks and L-frame buffer

`configs/train/default.yaml` exposes `eval_every` to skip mid-training
test eval (defaults to 1; set to 5 or 10 to halve wall-clock during dev).

## Outputs per run

`results/runs/{model}_{dataset}_fold{k}_{ts}/`:
- `metrics.json` — 10 keys: precision, recall, f1, accuracy, fpr, fnr,
  auc, latency_ms, params, flops
- `confusion.npy` — (4, 4) int64 confusion matrix
- `history.json` — per-epoch train+test metrics
- `model.pt` — final state_dict
- `config.json` — full resolved config bundle (data + model + train + args)
- `run.log` — full stdlib logging output
