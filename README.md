# MIDS — Detecting Tampering Attacks on CAN Bus with Bidirectional Mamba

This repository accompanies our paper:

> **MIDS: Detecting Tampering Attacks on CAN Bus with Bidirectional Mamba**
> *IEEE Transactions on Information Forensics and Security (under review).*

It contains both (i) the source code of the MIDS model and its ablations,
the original Tesla-dataset preprocessing pipeline used in Tables III–VI;
and (ii) a new, unified cross-dataset benchmark framework (`mids-bench/`)
used in Table VII of the journal revision.

---

## Repository layout

```
MIDS/
├── modules/          # MIDS model & ablation variants (PyTorch)
│   └── model.py        - MambaCAN_2Direction (= MIDS) + Sincos/Fre/LSTM/
│                         no-conv/no-pos/no-attn/no-id ablations + baselines
├── train/            # Original training & evaluation scripts (Tesla)
│   ├── train_mamba.py  - K-fold trainer for MIDS and all ablations
│   ├── mamba.py        - Mamba block utilities
│   ├── eval.py         - Standalone evaluation
│   └── Seo-train.py    - Reproduction of Seo et al. baseline
├── scripts/          # Tesla-dataset preprocessing pipeline
│   ├── own_data_process.py   - .asc → .csv
│   ├── modify_data.py        - Controlled tampering injection
│   ├── reshape.py            - Window grouping
│   ├── transfer.py           - .csv → .npy
│   ├── merge.py              - Dataset concatenation
│   ├── multi_data_process.py
│   ├── Seo_data_process.py   - Seo-dataset processing
│   └── data_visualizer.py
├── mids-bench/       # Unified cross-dataset benchmark (Table VII)
│   ├── src/
│   │   ├── datasets/   - Tesla / OTIDS / ROAD / CrySyS / CT&T parsers
│   │   ├── models/     - MIDS + 8 reproduced baselines
│   │   ├── train/      - Unified 5-fold trainer + metrics
│   │   └── utils/      - Seeding, logging, block-shuffled splits
│   ├── scripts/        - run_one.py, run_grid.py, make_table.py
│   ├── configs/        - per-model / per-dataset YAML
│   └── tests/          - Forward-pass smoke tests
├── MIDS_TIFS/        # Paper sources (LaTeX) and PDFs
│   ├── MIDS.tex        - Main manuscript
│   ├── reference.bib
│   ├── response.tex    - Point-by-point response to reviewers
│   ├── IEEEtran.cls
│   ├── *.png / *.jpg   - Figures
│   ├── MIDS.pdf
│   └── response.pdf
└── data/             # NOT committed — see datasets section below
```

## Datasets

Raw datasets are **not** included in this repository (see `.gitignore`).
You can obtain them from the original sources:

| Dataset | Source |
| --- | --- |
| Tesla (ours) | Released with this paper |
| OTIDS | Han Lab, Korea University |
| ROAD | Oak Ridge National Lab |
| CrySyS | CrySyS Lab, BME |
| CAN-Train-and-Test (CT&T) | Technical University of Denmark |

Place each dataset under `data/<name>/` and adjust the corresponding YAML in
`mids-bench/configs/data/` if needed.

## Quick start

### Reproducing Tables III–VI (Tesla, original pipeline)

```bash
# 1. Preprocess Tesla .asc → .npy
python scripts/own_data_process.py batch --input_dir data/raw --output_dir data/processed --output_file merged.csv
python scripts/modify_data.py --input_file data/processed/merged.csv --output_file data/modified/modified.csv --x 10 --modify_type Both
python scripts/reshape.py --input_dir data/modified --output_file data/reshaped/reshaped.csv --group_size 100
python scripts/transfer.py --input_csv data/reshaped/reshaped.csv --output_npy data/reshaped/reshaped.npy

# 2. Train MIDS (or any ablation variant) with 5-fold CV
python train/train_mamba.py
```

### Reproducing Table VII (cross-dataset benchmark)

```bash
cd mids-bench

# Run a single (model, dataset, fold)
python scripts/run_one.py --model mids --dataset otids --fold 0

# Run the full grid (9 models × 5 datasets × 5 folds)
python scripts/run_grid.py

# Compile the comparison table
python scripts/make_table.py
```

See `mids-bench/README.md` for full details.

## Requirements

- Python 3.10+, PyTorch 2.0+, CUDA 11.8+ (for `mamba-ssm`)
- See `mids-bench/requirements.txt` for the full benchmark stack
- `mamba-ssm` is **optional**: if it is not installed, the framework
  automatically falls back to a pure-PyTorch selective-scan implementation
  in `mids-bench/src/models/_mamba_torch.py`.

## License

MIT — see source files for individual headers.

## Citation

If you use this code, please cite our paper (BibTeX will be added once
publication is finalised).
