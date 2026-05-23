"""CrySyS — CAN traffic logs (Gazdag et al., Sci Data 2023).

Distribution layout (figshare collection release)::

    CrySyS dataset .../
        S-1-1/                                     # 26 recording subdirs
            S-1-1-benign.log                       # candump benign baseline
            S-1-1-benign.json                      # metadata
            S-1-1-benign-*.{pdf,png}               # plots (ignored)
            S-1-1-malicious-<strategy>-<id>-<sp>-<ep>.log
                                                   # benign + injected attack
            S-1-1-malicious-<strategy>-<id>-<sp>-<ep>-inj-messages.log
                                                   # pure attack frames subset
            S-1-1-malicious-<strategy>-<id>-<sp>-<ep>.json
                                                   # attack markers (start/end time, packet_ID)
        S-1-2/ ... T-3-2/

The .log files are standard candump format::

    (0.000000) can0 110#02202e1300181300

Attack windows are documented in the malicious .json files::

    markers: [
      {packet_ID: "0x410", time: 12.020789, description: "Start of the attack."},
      {packet_ID: "0x410", time: 18.091826, description: "End of the attack."},
    ]

A frame is an attack iff its absolute timestamp is in
[start_marker.time, end_marker.time] AND its CAN ID equals the
markers' packet_ID. The filename's strategy stem (``msg-inj`` vs
``msg-mod``) determines the attack class.

4-class label mapping
---------------------
====  ==============================  ============================
ID    Label                           Filename keyword
0     Normal                          benign.log; non-attack frames
1     Fabrication                     *-msg-inj-*  (injects new frames)
2     Masquerade                      *-msg-mod-*  (modifies existing)
3     (unused)                        —
====  ==============================  ============================

Attacks are sparse within 100-frame windows (the malicious log is
mostly the original benign traffic with a few injected/modified
frames during the attack interval), so this dataset uses
``WINDOW_LABEL_STRATEGY = "any-attack"``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
    _HAVE_TQDM = True
except ImportError:
    _HAVE_TQDM = False

from .base import CANDataset

logger = logging.getLogger(__name__)

# candump line regex, matches "(timestamp) can0 ID#DATA"
_LINE_RE = re.compile(
    rb"^\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-fR]*)"
)


class CrysysDataset(CANDataset):
    """CrySyS dataset (candump .log + attack-marker JSON)."""

    CLASS_NAMES = ("Normal", "Fabrication", "Masquerade", "Unused")

    # CrySyS attacks affect a few frames within the attack interval
    # (an injection adds a few new frames; a modification edits a few
    # existing frames). Per 100-frame window the attack frames are
    # always a small minority, so majority-vote would erase them.
    WINDOW_LABEL_STRATEGY = "any-attack"

    @property
    def name(self) -> str:
        return "crysys"

    # ------------------------------------------------------------------
    # Filename / class mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _attack_class_for_filename(stem: str) -> int:
        """``stem`` is the malicious-log basename without the .log."""
        lname = stem.lower()
        if "msg-inj" in lname or "msg_inj" in lname:
            return 1  # Fabrication
        if "msg-mod" in lname or "msg_mod" in lname:
            return 2  # Masquerade
        return 1  # safe fallback for unrecognised strategies

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _resolve_recording_dirs(self) -> List[Path]:
        """Return all per-recording subdirectories."""
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(f"CrySyS root {root} is not a directory.")
        # Recording dirs match prefixes S-* or T-*; tolerate one extra
        # nesting level if the release was unzipped into a wrapper folder.
        for d in (root, *sorted(root.iterdir())):
            if not d.is_dir():
                continue
            candidates = [
                p for p in d.iterdir()
                if p.is_dir() and (
                    p.name.startswith("S-") or p.name.startswith("T-")
                )
            ]
            if candidates:
                return sorted(candidates)
        raise FileNotFoundError(
            f"No 'S-*/' or 'T-*/' recording subdirs found under {root}. "
            f"Expected the unzipped CrySyS layout."
        )

    # ------------------------------------------------------------------
    # candump .log parsing (shared format with ROAD)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_log(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a candump .log into (frames[N,9] float32, timestamps[N] float64)."""
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
            return (
                np.empty((0, 9), dtype=np.float32),
                np.empty(0, dtype=np.float64),
            )
        id_arr = np.asarray(ids, dtype=np.float32)[:, None]
        payload_arr = (
            np.frombuffer(b"".join(payloads), dtype=np.uint8)
              .reshape(-1, 8)
              .astype(np.float32)
        )
        frames = np.concatenate([id_arr, payload_arr], axis=1)
        ts_arr = np.asarray(timestamps, dtype=np.float64)
        return frames, ts_arr

    # ------------------------------------------------------------------
    # Attack-marker resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_attack_window(
        json_path: Path,
    ) -> Optional[Tuple[float, float, int]]:
        """Return ``(start_time, end_time, target_id_int)`` or None.

        Pulls from the JSON ``markers`` list whose entries have
        ``packet_ID`` (hex string like "0x410"), ``time`` (float),
        and ``description`` (one of "Start of the attack" / "End of the
        attack"). If markers are absent or malformed, returns None.
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return None
        markers = meta.get("markers", [])
        if len(markers) < 2:
            return None

        start_time = None
        end_time = None
        target_id_hex = None
        for mk in markers:
            desc = (mk.get("description") or "").lower()
            t = mk.get("time")
            pid = mk.get("packet_ID")
            if t is None:
                continue
            if target_id_hex is None and pid:
                target_id_hex = pid
            if "start" in desc:
                start_time = float(t)
            elif "end" in desc:
                end_time = float(t)

        # Fallback: if descriptions weren't clean, take min/max by time.
        if start_time is None or end_time is None:
            times = [float(m["time"]) for m in markers if m.get("time") is not None]
            if len(times) < 2:
                return None
            start_time, end_time = min(times), max(times)
            target_id_hex = target_id_hex or markers[0].get("packet_ID")

        if target_id_hex is None:
            return None
        try:
            target_id = int(str(target_id_hex), 16)
        except (ValueError, TypeError):
            return None
        return start_time, end_time, target_id

    # ------------------------------------------------------------------
    # Public: _parse_raw produces (frames[N,9], labels[N])
    # ------------------------------------------------------------------

    def _label_malicious(
        self,
        log_path: Path,
        json_path: Path,
        frames: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        attack_cls = self._attack_class_for_filename(log_path.stem)
        labels = np.zeros(len(frames), dtype=np.int64)
        window = self._extract_attack_window(json_path)
        if window is None or len(frames) == 0:
            return labels

        start_t, end_t, target_id = window
        in_window = (timestamps >= start_t) & (timestamps <= end_t)
        id_match = frames[:, 0] == float(target_id)
        labels[in_window & id_match] = attack_cls
        return labels

    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        rec_dirs = self._resolve_recording_dirs()
        logger.info(
            "CrySyS: %d recording subdirs under %s",
            len(rec_dirs), self.root,
        )

        # Collect (log, json, role) triples. role is "benign" or "malicious".
        jobs: List[Tuple[Path, Optional[Path], str]] = []
        for d in rec_dirs:
            for log in sorted(d.glob("*.log")):
                if "inj-messages" in log.name:
                    continue  # pure-injection subset; not used directly
                json_path = log.with_suffix(".json")
                if "benign" in log.name.lower():
                    jobs.append((log, None, "benign"))
                else:
                    jobs.append(
                        (log, json_path if json_path.exists() else None,
                         "malicious"),
                    )
        logger.info(
            "CrySyS: %d log files to parse (~%.0f MB total)",
            len(jobs),
            sum(p.stat().st_size for p, _, _ in jobs) / 1024 / 1024,
        )

        per_class = [0, 0, 0, 0]
        all_frames: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        iterator = jobs
        pbar = None
        if _HAVE_TQDM:
            pbar = tqdm(jobs, desc="CrySyS parse", unit="log")
            iterator = pbar
        for log_path, json_path, role in iterator:
            try:
                frames, ts = self._parse_log(log_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  SKIP %s -> %s", log_path.name, exc)
                continue
            if role == "benign":
                labels = np.zeros(len(frames), dtype=np.int64)
            else:
                labels = self._label_malicious(
                    log_path, json_path, frames, ts,
                )
            for c in range(4):
                per_class[c] += int((labels == c).sum())
            if pbar is not None:
                pbar.set_postfix(dist=str(per_class))
            all_frames.append(frames)
            all_labels.append(labels)
        if pbar is not None:
            pbar.close()

        if not all_frames:
            raise RuntimeError("CrySyS: parsed 0 frames; check layout.")
        frames = np.concatenate(all_frames, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(
            "CrySyS: %d total frames; label dist = %s",
            len(frames),
            np.bincount(labels, minlength=self.num_classes).tolist(),
        )
        return frames, labels
