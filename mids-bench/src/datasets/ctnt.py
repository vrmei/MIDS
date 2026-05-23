"""CT&T — can-train-and-test (Lampe & Meng, VTC 2023).

CSV schema: timestamp, arbitration_id, data_field (packed hex), attack.
Layout: can-train-and-test/set_*/{train_01,test_01..04}/<type>-{1..4}.csv

4-class label mapping:
    0  Normal           attack-free, accessory
    1  DoS              DoS
    2  Fuzzy            fuzzing, double, triple, interval, systematic, speed
    3  Spoof            rpm, force-neutral, standstill
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

from .base import CANDataset

logger = logging.getLogger(__name__)


class CtntDataset(CANDataset):
    """can-train-and-test dataset, 4-class injection layout."""

    CLASS_NAMES = ("Normal", "DoS", "Fuzzy", "Spoof")

    # CT&T attacks are injected against captured benign traffic, so a
    # 100-frame window from any attack file typically contains only a
    # handful of attack frames mixed with mostly benign frames. Majority
    # vote collapses these to Normal and produces train/test splits
    # with ~0 attack windows. any-attack labels the window with its
    # attack class whenever at least one frame is anomalous.
    WINDOW_LABEL_STRATEGY = "any-attack"

    _ATTACK_KEYWORDS = (
        ("attack-free", 0), ("attack_free", 0), ("accessory", 0),
        ("force-neutral", 3), ("force_neutral", 3),
        ("standstill", 3), ("rpm", 3),
        ("fuzzing", 2), ("fuzzy", 2),
        ("double", 2), ("triple", 2), ("interval", 2),
        ("systematic", 2), ("speed", 2),
        ("dos", 1),
    )

    @property
    def name(self) -> str:
        return "ctnt"

    def _attack_class_for_filename(self, stem: str) -> int:
        lname = stem.lower()
        for needle, cls in self._ATTACK_KEYWORDS:
            if needle in lname:
                return cls
        return 1

    def _resolve_files(self) -> List[Path]:
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"CT&T root {root} is not a directory.")
        files = sorted(
            p for p in root.rglob("*.csv")
            if not p.name.startswith(".") and "__MACOSX" not in p.parts
        )
        if not files:
            raise FileNotFoundError(
                f"No .csv files found under {root}. Expected the "
                f"can-train-and-test layout: set_*/.../*.csv"
            )
        return files

    @staticmethod
    def _read_csv_any_encoding(
        path: Path, usecols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(path, encoding=enc, usecols=usecols)
            except UnicodeDecodeError:
                continue
            except ValueError:
                # usecols mismatch — retry without it.
                return pd.read_csv(path, encoding=enc)
        return pd.read_csv(path, encoding="latin-1", errors="replace")

    @staticmethod
    def _parse_data_field(series: pd.Series) -> np.ndarray:
        """Vectorised packed-hex -> (N, 8) float32. ~100x faster than
        per-row Python loop on multi-million-row CT&T CSVs."""
        s = series.astype(str).str.strip().str.upper()
        s = s.str.replace(r"^0X", "", regex=True)
        s = s.where(~s.isin(["", "NAN"]), "0" * 16)
        odd_mask = (s.str.len() % 2) == 1
        if odd_mask.any():
            s = s.mask(odd_mask, "0" + s)
        s = s.str.slice(0, 16).str.ljust(16, "0")
        valid = s.str.match(r"^[0-9A-F]{16}$")
        if not valid.all():
            s = s.where(valid, "0" * 16)
        joined = "".join(s.tolist())
        try:
            raw = bytes.fromhex(joined)
        except ValueError:
            arr = np.zeros((len(s), 8), dtype=np.float32)
            for i, x in enumerate(s):
                try:
                    arr[i] = np.frombuffer(bytes.fromhex(x), dtype=np.uint8)
                except ValueError:
                    pass
            return arr
        return (
            np.frombuffer(raw, dtype=np.uint8)
              .reshape(-1, 8)
              .astype(np.float32)
        )

    def _parse_csv(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        df = self._read_csv_any_encoding(
            path, usecols=["arbitration_id", "data_field", "attack"],
        )
        cols = {c.lower(): c for c in df.columns}

        id_col = (cols.get("arbitration_id") or cols.get("can_id")
                  or cols.get("id"))
        if id_col is None:
            raise ValueError(
                f"{path}: no arbitration_id/can_id/id column. "
                f"Columns: {list(df.columns)}"
            )
        ids = df[id_col]
        if ids.dtype == object:
            id_arr = np.array(
                [int(str(v).strip(), 16) for v in ids.tolist()],
                dtype=np.float32,
            )
        else:
            id_arr = ids.to_numpy(dtype=np.float32)

        data_col = (cols.get("data_field") or cols.get("data")
                    or cols.get("payload"))
        if data_col is not None:
            payload = self._parse_data_field(df[data_col].astype(str))
        else:
            byte_cols = [
                cols.get(f"byte{i}") or cols.get(f"b{i}") or cols.get(f"d{i}")
                for i in range(8)
            ]
            if not all(c is not None for c in byte_cols):
                raise ValueError(
                    f"{path}: no data_field column and missing per-byte "
                    f"columns. Columns: {list(df.columns)}"
                )
            payload = df[byte_cols].to_numpy(dtype=np.float32)

        frames = np.concatenate(
            [id_arr[:, None], payload], axis=1
        ).astype(np.float32)

        attack_col = (cols.get("attack") or cols.get("label")
                      or cols.get("class") or cols.get("is_attack"))
        attack_cls = self._attack_class_for_filename(path.stem)
        if attack_col is None:
            labels = np.full(len(df), attack_cls, dtype=np.int64)
        else:
            flag = df[attack_col].to_numpy(dtype=np.int64)
            labels = np.where(flag > 0, attack_cls, 0).astype(np.int64)
        return frames, labels

    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        files = self._resolve_files()
        total_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
        logger.info(
            "CT&T: parsing %d CSV files (%.0f MB total) under %s",
            len(files), total_mb, self.root,
        )
        logger.info(
            "      First-time parse may take 5-10 min on local SSD; "
            "subsequent runs hit cache."
        )

        all_frames: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        per_class = [0, 0, 0, 0]
        iterator = files
        pbar = None
        if _HAVE_TQDM:
            pbar = tqdm(files, desc="CT&T parse", unit="file")
            iterator = pbar
        for path in iterator:
            try:
                frames, labels = self._parse_csv(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  SKIP %s -> %s", path.name, exc)
                continue
            for c in range(4):
                per_class[c] += int((labels == c).sum())
            if pbar is not None:
                pbar.set_postfix(dist=str(per_class))
            all_frames.append(frames)
            all_labels.append(labels)
        if pbar is not None:
            pbar.close()

        if not all_frames:
            raise RuntimeError("CT&T: parsed 0 frames; check layout.")
        frames = np.concatenate(all_frames, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(
            "CT&T: %d total frames; label dist = %s",
            len(frames),
            np.bincount(labels, minlength=self.num_classes).tolist(),
        )
        return frames, labels
