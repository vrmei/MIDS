"""OTIDS — HCRL Hyundai Sonata CAN intrusion dataset (Lee et al. 2017).

Original distribution
---------------------
HCRL releases OTIDS as four plain-text logs::

    Attack_free_dataset.txt        # benign-only
    DoS_attack_dataset.txt         # DoS injection
    Fuzzy_attack_dataset.txt       # fuzzy/random ID injection
    Impersonation_attack_dataset.txt  # impersonation (a.k.a. spoofing)

Each line is::

    Timestamp: 1478198376.389266        ID: 0316    000    DLC: 8    05 21 68 09 21 21 00 6f

Sometimes the data bytes section ends with ``R`` for remote frames; we
treat those as 0-byte payloads zero-padded to 8.

Label space (4-class injection layout)
--------------------------------------
=== ===================  ============================
ID  Label                Source file
0   Normal               Attack_free_dataset.txt
1   DoS                  DoS_attack_dataset.txt
2   Fuzzy                Fuzzy_attack_dataset.txt
3   Impersonation/Spoof  Impersonation_attack_dataset.txt
=== ===================  ============================

Note that the attack files contain a *mixture* of benign + attack frames
(the attack tool injected onto a live bus). HCRL does not provide
per-frame attack masks, so this loader uses the conservative
"file-as-label" convention shared by most published reproductions:
all frames in DoS_attack_dataset.txt are labelled DoS, etc. This
inflates the apparent attack frequency relative to ground truth, but
the windowed majority-vote step in the base class collapses
attack-dominant windows into the right label, so downstream metrics
remain meaningful.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .base import CANDataset

logger = logging.getLogger(__name__)


class OtidsDataset(CANDataset):
    """OTIDS dataset, 4-class injection layout."""

    CLASS_NAMES = ("Normal", "DoS", "Fuzzy", "Spoof")

    # Filename -> class id. Lower-cased substring match for robustness
    # since HCRL has shipped slightly different filenames over time.
    _FILE_LABELS = (
        ("attack_free", 0),
        ("dos", 1),
        ("fuzzy", 2),
        ("impersonation", 3),
        ("spoof", 3),
    )

    _LINE_RE = re.compile(
        r"Timestamp:\s*(?P<ts>\S+)\s+"
        r"ID:\s*(?P<id>\S+)\s+\S+\s+"   # the literal "000" between ID and DLC
        r"DLC:\s*(?P<dlc>\d+)\s*"
        r"(?P<bytes>.*)$"
    )

    @property
    def name(self) -> str:
        return "otids"

    def _resolve_files(self) -> List[Tuple[Path, int]]:
        root = Path(self.root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"OTIDS root {root} is not a directory. Expected the four "
                f"HCRL .txt files inside."
            )
        out: List[Tuple[Path, int]] = []
        for path in sorted(root.iterdir()):
            if path.suffix.lower() not in {".txt", ".log", ".csv"}:
                continue
            stem = path.stem.lower()
            for needle, label in self._FILE_LABELS:
                if needle in stem:
                    out.append((path, label))
                    break
        if not out:
            raise FileNotFoundError(
                f"No OTIDS .txt files found under {root}. Looking for "
                f"filenames containing any of {[n for n, _ in self._FILE_LABELS]}."
            )
        return out

    def _parse_one_line(self, line: str) -> Tuple[int, np.ndarray] | None:
        match = self._LINE_RE.match(line.strip())
        if not match:
            return None
        try:
            can_id = int(match["id"], 16)
        except ValueError:
            return None
        dlc = int(match["dlc"])
        byte_tokens = match["bytes"].split()
        # Skip remote frames or malformed lines.
        if dlc == 0 or any(t.upper() == "R" for t in byte_tokens):
            payload = np.zeros(8, dtype=np.float32)
        else:
            try:
                vals = [int(t, 16) for t in byte_tokens[:8]]
            except ValueError:
                return None
            payload = np.zeros(8, dtype=np.float32)
            payload[: len(vals)] = vals
        return can_id, payload

    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        files = self._resolve_files()
        logger.info("OTIDS: parsing %d files under %s", len(files), self.root)

        all_frames: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        for path, label in files:
            ids: List[int] = []
            payloads: List[np.ndarray] = []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parsed = self._parse_one_line(line)
                    if parsed is None:
                        continue
                    can_id, payload = parsed
                    ids.append(can_id)
                    payloads.append(payload)
            if not ids:
                logger.warning("OTIDS: no frames parsed from %s", path)
                continue
            id_arr = np.asarray(ids, dtype=np.float32)[:, None]
            payload_arr = np.stack(payloads, axis=0)
            frames = np.concatenate([id_arr, payload_arr], axis=1)
            labels = np.full(frames.shape[0], label, dtype=np.int64)
            logger.info(
                "  %s: %d frames, label=%d (%s)",
                path.name, frames.shape[0], label, self.CLASS_NAMES[label],
            )
            all_frames.append(frames)
            all_labels.append(labels)

        frames = np.concatenate(all_frames, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        logger.info(
            "OTIDS: parsed %d total frames; label dist = %s",
            len(frames),
            np.bincount(labels, minlength=self.num_classes).tolist(),
        )
        return frames, labels
