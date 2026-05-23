"""ROAD — Real ORNL Automotive Dynamometer dataset (Hollifield et al. 2021).

Frame data lives in `.log` files (can-utils candump format):
    (1110000000.000000) can0 0F4#960C010204B10240

Attack windows are documented in `attacks/capture_metadata.json`.
A frame is an attack iff its relative time is in `injection_interval`
AND its ID matches `injection_id`. Accelerator attacks have a null
interval, so the whole capture is labelled as the attack class.

4-class label mapping (consistent with Tesla):
    0  Normal
    1  Vehicle-state (accelerator_attack_*)
    2  Data-tamper (correlated_signal, max_*, reverse_light)
    3  Mixed (fuzzing_*)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .base import CANDataset

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(
    rb"^\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-fR]*)"
)


class RoadDataset(CANDataset):
    """ROAD dataset (raw candump .log files), mapped to 4 classes."""

    CLASS_NAMES = ("Normal", "Vehicle-state", "Data-tamper", "Mixed")

    # ROAD's flam injection puts ~1 attack frame between adjacent
    # legitimate frames of the same ID, so a 100-frame window only
    # contains a handful of attack frames mixed into mostly normal
    # traffic. Majority vote would label every such window Normal and
    # zero out the attack classes; any-attack labels the window with
    # the actual attack class whenever any frame inside is anomalous.
    WINDOW_LABEL_STRATEGY = "any-attack"

    _ATTACK_KEYWORDS = (
        ("accelerator_attack", 1),
        ("correlated_signal", 2),
        ("max_speedometer", 2),
        ("max_engine", 2),
        ("max_rpm", 2),
        ("reverse_light", 2),
        ("fuzzing", 3),
    )

    @property
    def name(self) -> str:
        return "road"

    def _attack_class_for_filename(self, name: str) -> int:
        lname = name.lower()
        for needle, cls in self._ATTACK_KEYWORDS:
            if needle in lname:
                return cls
        return 1

    def _resolve_logs(self) -> Tuple[List[Path], List[Path]]:
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"ROAD root {root} is not a directory.")
        for d in (root, root / "road"):
            if (d / "ambient").is_dir() and (d / "attacks").is_dir():
                ambient_dir = d / "ambient"
                attacks_dir = d / "attacks"
                break
        else:
            raise FileNotFoundError(
                f"Could not find 'ambient/' + 'attacks/' under {root} or "
                f"{root / 'road'}."
            )

        def logs(d: Path) -> List[Path]:
            return sorted(p for p in d.glob("*.log") if not p.name.startswith("."))

        return logs(ambient_dir), logs(attacks_dir)

    def _load_attack_metadata(self) -> Dict[str, dict]:
        root = Path(self.root)
        for cand in (
            root / "attacks" / "capture_metadata.json",
            root / "road" / "attacks" / "capture_metadata.json",
        ):
            if cand.exists():
                with open(cand, "r", encoding="utf-8") as f:
                    return json.load(f)
        logger.warning(
            "ROAD: no attacks/capture_metadata.json found; falling back to "
            "file-as-label (every frame in an attack log gets per-file class)."
        )
        return {}

    @staticmethod
    def _parse_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        ids: List[int] = []
        payloads: List[bytes] = []
        timestamps: List[float] = []
        with open(path, "rb") as f:
            for raw in f:
                m = _LINE_RE.match(raw)
                if not m:
                    continue
                ts = float(m.group(1))
                can_id = int(m.group(2), 16)
                data = m.group(3)
                if data.startswith(b"R") or len(data) == 0:
                    pl = b"\x00" * 8
                else:
                    if len(data) % 2 == 1:
                        data = data + b"0"
                    if len(data) >= 16:
                        data = data[:16]
                    else:
                        data = data + b"0" * (16 - len(data))
                    pl = bytes.fromhex(data.decode("ascii", errors="replace"))
                timestamps.append(ts)
                ids.append(can_id)
                payloads.append(pl)
        if not ids:
            return (np.empty((0, 9), dtype=np.float32),
                    np.empty(0, dtype=np.float64))
        id_arr = np.asarray(ids, dtype=np.float32)[:, None]
        payload_arr = np.frombuffer(b"".join(payloads), dtype=np.uint8).reshape(-1, 8).astype(np.float32)
        frames = np.concatenate([id_arr, payload_arr], axis=1)
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        return frames, ts_arr

    def _label_attack_frames(
        self,
        path: Path,
        frames: np.ndarray,
        timestamps: np.ndarray,
        metadata: Dict[str, dict],
    ) -> np.ndarray:
        attack_cls = self._attack_class_for_filename(path.stem)
        meta = metadata.get(path.stem)
        labels = np.zeros(len(frames), dtype=np.int64)

        if meta is None or meta.get("injection_interval") is None:
            labels[:] = attack_cls
            return labels

        interval = meta["injection_interval"]
        t0 = timestamps[0] if len(timestamps) else 0.0
        rel_t = timestamps - t0
        in_window = (rel_t >= interval[0]) & (rel_t <= interval[1])

        injection_id_hex = meta.get("injection_id")
        target_id = None
        if injection_id_hex:
            try:
                target_id = float(int(str(injection_id_hex), 16))
            except (ValueError, TypeError):
                target_id = None

        if target_id is not None:
            id_match = frames[:, 0] == target_id
            attack_mask = in_window & id_match
        else:
            attack_mask = in_window

        labels[attack_mask] = attack_cls
        return labels

    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        ambient_logs, attack_logs = self._resolve_logs()
        attack_meta = self._load_attack_metadata()
        logger.info(
            "ROAD: parsing %d ambient + %d attack .log files",
            len(ambient_logs), len(attack_logs),
        )

        all_frames: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for path in ambient_logs:
            frames, _ = self._parse_log(path)
            labels = np.zeros(len(frames), dtype=np.int64)
            logger.info("  ambient  %-50s %d frames", path.name, len(frames))
            all_frames.append(frames)
            all_labels.append(labels)

        for path in attack_logs:
            frames, ts = self._parse_log(path)
            labels = self._label_attack_frames(path, frames, ts, attack_meta)
            n_attack = int((labels > 0).sum())
            cls = self._attack_class_for_filename(path.stem)
            logger.info(
                "  attack   %-50s %d frames, %d attack (cls=%d %s)",
                path.name, len(frames), n_attack, cls, self.CLASS_NAMES[cls],
            )
            all_frames.append(frames)
            all_labels.append(labels)

        frames = np.concatenate(all_frames, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(
            "ROAD: %d total frames; label dist = %s",
            len(frames),
            np.bincount(labels, minlength=self.num_classes).tolist(),
        )
        return frames, labels
