"""Tesla Model 3 CAN dataset.

Two ingestion paths are supported:

* **Pre-windowed legacy ``.npy``.** The original MIDS repo's
  ``transfer.py`` produces an ``all.npy`` of shape
  ``(N_windows, L * F + 1)``: each row is a flattened window followed
  by its integer label. If ``root`` points at a ``.npy`` file (or a
  directory containing ``all.npy`` / ``merged/all.npy``), this path is
  used and ``_parse_raw`` is never called. Useful for matching the
  exact 96.94% F1 in the paper without re-running preprocessing.

  Crucially, this path returns a memory-mapped array via
  :meth:`_load_flat_legacy`, so the multi-GB source file is *not*
  duplicated to ``cache_dir/`` — only the small folds.npz lands there.

* **Per-frame CSV directory.** ``root`` points at a directory of
  ``.csv`` files written by ``own_data_process.py`` +
  ``modify_data.py``: columns are
  ``timestamp, ID, DLC, byte0, ..., byte7, label``. Files are
  concatenated in sorted order to give the per-frame stream. This path
  *does* write a windows.npy + labels.npy cache because the parsing
  step is expensive.

The first path matching the on-disk layout wins. Both feed into the
base class's block-shuffled fold logic, so the protocol is identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import CANDataset

logger = logging.getLogger(__name__)


class TeslaDataset(CANDataset):
    """Tesla Model 3 dataset with 4 classes: Normal / ID / Data / Both."""

    @property
    def name(self) -> str:
        return "tesla"

    # ------------------------------------------------------------------
    # Pre-windowed legacy .npy path
    # ------------------------------------------------------------------

    def _resolve_legacy_npy(self) -> Optional[Path]:
        """Return the legacy ``all.npy`` path if one is reachable."""
        # Re-cast unconditionally so a string self.root still works.
        root = Path(self.root)
        if root.is_file() and root.suffix == ".npy":
            return root
        if not root.is_dir():
            logger.info(
                "Tesla root %s is not a directory; will try CSV parse path.",
                root,
            )
            return None
        for sub in ("all.npy", "merged/all.npy", "owndata/merged/all.npy"):
            candidate = root / sub
            if candidate.exists():
                return candidate
        logger.info(
            "No legacy all.npy found under %s (tried %s); will try CSV parse.",
            root,
            ["all.npy", "merged/all.npy", "owndata/merged/all.npy"],
        )
        return None

    def _load_flat_legacy(self) -> Optional[np.ndarray]:
        """Memory-map the legacy ``(N, L*F + 1)`` flat array if present.

        Returning the mmap'd array lets the base class skip the
        windows/labels disk cache — important for multi-GB datasets
        where duplicating to ``cache_dir/`` would blow out the disk.
        """
        npy_path = self._resolve_legacy_npy()
        if npy_path is None:
            return None
        logger.info("Memory-mapping legacy flat .npy from %s", npy_path)
        return np.load(npy_path, mmap_mode="r")

    # ------------------------------------------------------------------
    # Per-frame CSV path
    # ------------------------------------------------------------------

    _CSV_REQUIRED_COLS = (
        "ID",
        "byte0", "byte1", "byte2", "byte3",
        "byte4", "byte5", "byte6", "byte7",
        "label",
    )

    def _resolve_csv_files(self) -> List[Path]:
        if self.root.is_file() and self.root.suffix == ".csv":
            return [self.root]
        if not self.root.is_dir():
            return []
        return sorted(self.root.rglob("*.csv"))

    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        csv_files = self._resolve_csv_files()
        if not csv_files:
            raise FileNotFoundError(
                f"No legacy .npy and no .csv files found under {self.root}. "
                f"Expected either an all.npy from the legacy pipeline or a "
                f"directory of CSVs with columns {self._CSV_REQUIRED_COLS}."
            )

        logger.info("Parsing %d CSV recording(s) under %s",
                    len(csv_files), self.root)

        frames_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        for path in csv_files:
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                df = pd.read_csv(path, encoding="latin-1", errors="replace")
            missing = [c for c in self._CSV_REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{path}: missing required columns {missing}. "
                    f"Found {list(df.columns)}."
                )
            # ID may come as hex string "0x102" or int.
            ids = df["ID"]
            if ids.dtype == object:
                id_arr = np.array(
                    [int(str(v), 0) for v in ids.tolist()],
                    dtype=np.int64,
                )
            else:
                id_arr = ids.to_numpy(dtype=np.int64)

            payload = df[
                [f"byte{i}" for i in range(8)]
            ].to_numpy(dtype=np.float32)

            # frames: (N, 9) = [ID, byte0..byte7]
            recording = np.concatenate(
                [id_arr.astype(np.float32)[:, None], payload], axis=1
            )
            frames_list.append(recording)
            labels_list.append(df["label"].to_numpy(dtype=np.int64))

        frames = np.concatenate(frames_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        logger.info(
            "Parsed %d total frames; label distribution = %s",
            len(frames),
            np.bincount(labels, minlength=self.num_classes).tolist(),
        )
        return frames, labels
