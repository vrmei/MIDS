#!/usr/bin/env bash
# ============================================================
# Tesla CAN dataset end-to-end preprocessing pipeline
# ============================================================
# Chains the four stages used to produce the .npy training
# tensors for Tables III–VI in the paper:
#
#   1. own_data_process.py   .asc  → per-file .csv
#   2. modify_data.py        .csv  → tampered .csv (CANID / payload / Both)
#   3. reshape.py            many .csv → windowed .csv (group_size = 100)
#   4. merge.py              per-scenario windowed .csv → merged .csv
#   5. transfer.py           merged .csv → .npy
#
# Usage:
#   bash scripts/process_tesla.sh                       # process default scenarios
#   bash scripts/process_tesla.sh high-speed standby    # process given scenarios
#
# Expected raw layout:
#   data/owndata/origin/<scenario>/*.asc
#
# Output layout:
#   data/owndata/processed/<scenario>/*_processed.csv
#   data/owndata/attackdata/<scenario>/<scenario>-<modtype>-<x>.csv
#   data/owndata/reshape/<scenario>/<scenario>.csv
#   data/owndata/merged/<scenario>_merged.csv
#   data/owndata/merged/<scenario>_merged.npy   <-- consumed by train/train_mamba.py
# ============================================================

set -euo pipefail

# ---- Configuration --------------------------------------------------------
PY="${PY:-python}"                            # override with: PY=python3
GROUP_SIZE="${GROUP_SIZE:-100}"               # window length used in the paper
INJECTION_RATIOS=(${INJECTION_RATIOS:-2 5 10})  # one attack per N messages
MOD_TYPES=(${MOD_TYPES:-CANID payload Both})    # three tampering types

# Default scenarios if none supplied on the command line
DEFAULT_SCENARIOS=(high-speed standby)
SCENARIOS=("${@:-${DEFAULT_SCENARIOS[@]}}")

DATA_ROOT="data/owndata"
ORIGIN_DIR="${DATA_ROOT}/origin"
PROC_DIR="${DATA_ROOT}/processed"
ATK_DIR="${DATA_ROOT}/attackdata"
RESHAPE_DIR="${DATA_ROOT}/reshape"
MERGED_DIR="${DATA_ROOT}/merged"

mkdir -p "${PROC_DIR}" "${ATK_DIR}" "${RESHAPE_DIR}" "${MERGED_DIR}"

# ---- Helpers --------------------------------------------------------------
log() { printf "\n[\033[1;36m%s\033[0m] %s\n" "$(date +%H:%M:%S)" "$*"; }

# ---- Main loop ------------------------------------------------------------
for scenario in "${SCENARIOS[@]}"; do
    log "===== Scenario: ${scenario} ====="

    in_dir="${ORIGIN_DIR}/${scenario}"
    if [[ ! -d "${in_dir}" ]]; then
        echo "  Skipping: ${in_dir} does not exist."
        continue
    fi

    # ---- Stage 1: .asc → .csv (one CSV per .asc) --------------------------
    proc_dir="${PROC_DIR}/${scenario}"
    log "Stage 1/5  own_data_process.py  (.asc → .csv)"
    "${PY}" scripts/own_data_process.py batch_single \
        --input_dir  "${in_dir}" \
        --output_dir "${proc_dir}"

    # ---- Stage 2: inject tampering attacks --------------------------------
    atk_dir="${ATK_DIR}/${scenario}"
    mkdir -p "${atk_dir}"
    log "Stage 2/5  modify_data.py  (inject tampering)"
    for src in "${proc_dir}"/*_processed.csv; do
        [[ -e "${src}" ]] || { echo "  No processed files for ${scenario}, skipping."; break; }
        base="$(basename "${src}" _processed.csv)"
        for mod in "${MOD_TYPES[@]}"; do
            for x in "${INJECTION_RATIOS[@]}"; do
                out="${atk_dir}/${base}-${mod}-${x}.csv"
                if [[ -f "${out}" ]]; then
                    echo "    [skip] ${out}"
                    continue
                fi
                "${PY}" scripts/modify_data.py \
                    --input_file  "${src}" \
                    --output_file "${out}" \
                    --x "${x}" \
                    --modify_type "${mod}"
            done
        done
    done

    # ---- Stage 3: window grouping -----------------------------------------
    reshape_out="${RESHAPE_DIR}/${scenario}/${scenario}.csv"
    mkdir -p "$(dirname "${reshape_out}")"
    log "Stage 3/5  reshape.py  (group_size=${GROUP_SIZE})"
    "${PY}" scripts/reshape.py \
        --input_dir  "${atk_dir}" \
        --output_file "${reshape_out}" \
        --group_size "${GROUP_SIZE}"

    # ---- Stage 4: per-scenario merge --------------------------------------
    merged_csv="${MERGED_DIR}/${scenario}_merged.csv"
    log "Stage 4/5  merge.py  (concat windowed CSVs)"
    "${PY}" scripts/merge.py \
        --input_dir  "$(dirname "${reshape_out}")" \
        --output_file "${merged_csv}"

    # ---- Stage 5: CSV → NPY -----------------------------------------------
    merged_npy="${MERGED_DIR}/${scenario}_merged.npy"
    log "Stage 5/5  transfer.py  (.csv → .npy)"
    "${PY}" - <<PY_EOF
import sys
sys.path.insert(0, "scripts")
from transfer import csv_to_npy
csv_to_npy("${merged_csv}", "${merged_npy}")
PY_EOF

    log "Done: ${merged_npy}"
done

log "All scenarios processed."
