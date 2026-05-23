"""Abstract base class for CAN intrusion-detection datasets.

A concrete subclass is expected to implement :meth:`_parse_raw`, which
returns the full per-frame stream as ``(frames, labels)`` numpy arrays.
The base class then handles:

* Sliding-window construction with stride ``S = L`` (no overlap).
* Block-shuffled fold generation per §VI-A4 of the MIDS paper.
* Disk caching, keyed on the dataset name + split parameters, so that
  the heavy raw parsing is paid exactly once per dataset.

Subclasses can also short-circuit ``_parse_raw`` by overriding
:meth:`_load_prewindowed`, which is useful when the raw data has
already been windowed by a legacy pipeline (e.g., the existing
MIDS repo's ``transfer.py`` produces an ``all.npy`` of shape
``(N, L*F + 1)`` where the trailing column is the label).
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.splits import (
    FoldIndices,
    build_block_shuffled_folds,
    load_folds,
    save_folds,
)

logger = logging.getLogger(__name__)


class CANDataset(Dataset, ABC):
    """Base class for windowed CAN-frame datasets.

    Attributes:
        L: Window length in frames.
        S: Sliding-window stride (= L, so windows are non-overlapping).
        F: Per-frame feature count (1 ID + 8 payload bytes by default).
    """

    L: int = 100
    S: int = 100
    F: int = 9

    #: How to assign a single label to a 100-frame window from per-frame
    #: labels. Two policies, picked per-dataset:
    #:
    #: ``"majority"`` — argmax of the frame-label histogram. Right for
    #:   datasets where attacks dominate the window when present
    #:   (Tesla: I in {2,5,...,100}; OTIDS: file-as-label).
    #: ``"any-attack"`` — if any frame in the window has nonzero label,
    #:   the window inherits that label (mode of the nonzero labels).
    #:   Right for datasets with sparse attack frames (ROAD: flam
    #:   injection puts ~1 attack frame per window).
    WINDOW_LABEL_STRATEGY: str = "majority"

    def __init__(
        self,
        root: str,
        split: str,
        fold: int,
        n_folds: int = 5,
        num_classes: int = 4,
        cache_dir: str = "cache/",
        seed: int = 42,
        chunk_size: int = 1000,
        buffer_windows: int = 100,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not (0 <= fold < n_folds):
            raise ValueError(f"fold {fold} out of range [0, {n_folds})")

        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.n_folds = n_folds
        self.num_classes = num_classes
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.chunk_size = chunk_size
        self.buffer_windows = buffer_windows

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._windows, self._labels, self._fold_idx = self._load_or_build()

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    @abstractmethod
    def _parse_raw(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parse the dataset's raw form into a per-frame stream.

        Returns:
            ``(frames, labels)`` where ``frames`` has shape ``(N, F)`` and
            ``labels`` has shape ``(N,)`` with integer values in
            ``[0, num_classes)``. ``frames[:, 0]`` must be the CAN ID
            (integer-valued, but stored as float is fine), and
            ``frames[:, 1:]`` the payload bytes in canonical order.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, filesystem-safe identifier (e.g., ``"tesla"``)."""

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def _load_prewindowed(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(windows, labels)`` if a pre-windowed cache exists.

        Default implementation returns ``None``, telling the base class
        to fall back to :meth:`_parse_raw` + sliding-window construction.

        When overridden and a non-None tuple is returned, ``windows``
        must have shape ``(N_windows, L, F)`` and ``labels`` shape
        ``(N_windows,)``.
        """
        return None

    def _load_flat_legacy(self) -> Optional[np.ndarray]:
        """Return an mmap'd ``(N, L*F + 1)`` flat legacy array, or None.

        Datasets whose authoritative form on disk is already a flat
        ``.npy`` file (e.g., the Tesla pipeline's ``transfer.py`` output)
        should override this to return ``np.load(path, mmap_mode='r')``.
        When non-None is returned, the base class skips the windows/
        labels disk cache entirely — the source file IS the cache, and
        slicing is done lazily through the memory map. This avoids
        duplicating multi-gigabyte datasets to ``cache_dir/``.
        """
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_key(self) -> str:
        """Stable hash of the parameters that affect the cached arrays."""
        h = hashlib.sha1()
        for piece in (
            self.name,
            self.L,
            self.S,
            self.F,
            self.n_folds,
            self.num_classes,
            self.chunk_size,
            self.buffer_windows,
            self.seed,
            # Strategy is hashed so changing the labelling rule
            # auto-invalidates the windowed cache.
            self.WINDOW_LABEL_STRATEGY,
        ):
            h.update(repr(piece).encode())
        return h.hexdigest()[:12]

    def _load_or_build(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, FoldIndices]:
        """Resolve windows + labels + folds for the current split.

        Three code paths, in priority order:

        1. **Flat legacy ``.npy``** (``_load_flat_legacy`` returns array):
           the source file is treated as the cache. Only the small
           folds.npz is written under ``cache_dir/``. Slicing uses the
           memory map so we never materialise the full dataset.

        2. **Pre-windowed ``(windows, labels)``** (``_load_prewindowed``
           returns tuple): same as case 3 below, but skip the parse step.

        3. **Raw frames via** ``_parse_raw``: parse, sliding-window,
           write windows.npy + labels.npy to ``cache_dir`` so
           subsequent runs skip the parse.
        """
        key = self._cache_key()
        folds_path = self.cache_dir / f"{self.name}_{key}_folds.npz"

        # ----- Case 1: flat legacy npy, zero-copy ------------------
        flat = self._load_flat_legacy()
        if flat is not None:
            expected_cols = self.L * self.F + 1
            if flat.ndim != 2 or flat.shape[1] != expected_cols:
                raise ValueError(
                    f"Flat legacy array must be (N, {expected_cols}); "
                    f"got {flat.shape}"
                )
            n_windows = flat.shape[0]
            folds = self._load_or_build_folds(folds_path, n_windows)

            fold_idx = folds[self.fold]
            idx = fold_idx.train if self.split == "train" else fold_idx.test
            sliced = np.ascontiguousarray(flat[idx])  # (n_split, L*F+1)
            windows_slice = sliced[:, :-1].reshape(-1, self.L, self.F).astype(
                np.float32, copy=False
            )
            labels_slice = sliced[:, -1].astype(np.int64, copy=False)

            windows_tensor = torch.from_numpy(windows_slice).float()
            labels_tensor = torch.from_numpy(labels_slice).long()
            return windows_tensor, labels_tensor, fold_idx

        # ----- Case 2 / 3: cached windows + labels -----------------
        windows_path = self.cache_dir / f"{self.name}_{key}_windows.npy"
        labels_path = self.cache_dir / f"{self.name}_{key}_labels.npy"

        if windows_path.exists() and labels_path.exists():
            sz_mb = windows_path.stat().st_size / 1024 / 1024
            logger.info(
                "Cache hit: %s (%.0f MB; mmap + chunked slicing)",
                windows_path, sz_mb,
            )
            # mmap to keep peak RAM low: full-load was 7+ GB for big
            # datasets like CrySyS/CT&T, and combined with the slice
            # copy below it OOM'd 16 GB-RAM machines. The chunked-
            # indexing path further down reads contiguous batches from
            # the mmap, so the previous "random-page-fault" slowness
            # is avoided.
            windows = np.load(windows_path, mmap_mode="r")
            labels = np.load(labels_path)
        else:
            logger.info("Cache miss for key=%s; building...", key)
            windows, labels = self._build_windows()
            sz_gb = windows.nbytes / 1024 ** 3
            logger.info(
                "Writing windowed cache (%.1f GB) -> %s",
                sz_gb, windows_path,
            )
            np.save(windows_path, windows)
            np.save(labels_path, labels)
            logger.info(
                "Cache populated: %d windows, key=%s",
                len(windows),
                key,
            )

        folds = self._load_or_build_folds(folds_path, len(windows))
        fold_idx = folds[self.fold]
        idx = fold_idx.train if self.split == "train" else fold_idx.test

        # Materialise the split via chunked fancy-indexing. Two reasons
        # over the previous one-shot `windows[idx]`:
        #   1. Pre-allocates the output, so peak RAM = output (~6 GB)
        #      rather than output + a transient `_ArrayMemoryError`-
        #      sized intermediate (~13 GB on CT&T/CrySyS).
        #   2. When `windows` is mmap'd, this turns N million random
        #      page-faults into a sequence of ~50k-window block reads,
        #      ~50x faster on Windows than one giant fancy index.
        n_split = len(idx)
        out_shape = (n_split, self.L, self.F)
        out_gb = (n_split * self.L * self.F * 4) / 1024 ** 3
        logger.info(
            "Materialising %s split (%d windows, %.2f GB) via chunked reads",
            self.split, n_split, out_gb,
        )
        windows_slice = np.empty(out_shape, dtype=np.float32)
        CHUNK = 50_000  # ~180 MB per chunk at (50k, 100, 9) float32
        for start in range(0, n_split, CHUNK):
            end = min(start + CHUNK, n_split)
            windows_slice[start:end] = windows[idx[start:end]]
        labels_slice = np.ascontiguousarray(labels[idx])

        windows_tensor = torch.from_numpy(windows_slice).float()
        labels_tensor = torch.from_numpy(labels_slice).long()
        return windows_tensor, labels_tensor, fold_idx

    def _load_or_build_folds(
        self, folds_path: Path, n_windows: int
    ) -> "list[FoldIndices]":
        """Cached fold-index resolution. Tiny — always safe to write."""
        if folds_path.exists():
            return load_folds(str(folds_path), self.n_folds)
        folds = build_block_shuffled_folds(
            n_windows=n_windows,
            n_folds=self.n_folds,
            chunk_size=self.chunk_size,
            buffer_windows=self.buffer_windows,
            seed=self.seed,
        )
        save_folds(folds, str(folds_path))
        logger.info(
            "Folds written: %d folds across %d windows -> %s",
            self.n_folds, n_windows, folds_path,
        )
        return folds

    def _build_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the windowed tensor from either pre-windowed cache or raw."""
        prewin = self._load_prewindowed()
        if prewin is not None:
            windows, labels = prewin
            self._sanity_check_windows(windows, labels)
            return windows.astype(np.float32, copy=False), labels.astype(
                np.int64, copy=False
            )

        frames, frame_labels = self._parse_raw()
        return self._window_stream(frames, frame_labels)

    def _window_stream(
        self, frames: np.ndarray, frame_labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Carve a per-frame stream into non-overlapping windows.

        Per-window label depends on :attr:`WINDOW_LABEL_STRATEGY`:

        ``"majority"`` — argmax of the per-frame label histogram. Right
            for datasets where attacks dominate the window when
            present (Tesla I in {2,5,...,100}, OTIDS file-as-label).

        ``"any-attack"`` — if the window contains any nonzero frame
            label, take the **mode of the nonzero labels**; else 0.
            Right for datasets where attack frames are extremely
            sparse within a window (ROAD's flam injection produces
            ~1 attack frame per 100-frame window).
        """
        if frames.ndim != 2 or frames.shape[1] != self.F:
            raise ValueError(
                f"frames must be (N, {self.F}); got {frames.shape}"
            )
        if frame_labels.shape != (frames.shape[0],):
            raise ValueError(
                f"frame_labels shape {frame_labels.shape} doesn't match "
                f"frames {frames.shape}"
            )

        n_full = frames.shape[0] // self.L
        if n_full == 0:
            raise ValueError(
                f"Need at least L={self.L} frames; got {frames.shape[0]}"
            )
        truncated = n_full * self.L
        frames = frames[:truncated]
        frame_labels = frame_labels[:truncated]

        # Avoid the implicit memory copy here. `frames` comes out of
        # `_parse_raw` already as float32 in every dataset parser, so
        # `astype(copy=False)` is a no-op rather than a 7 GB
        # allocation. Skipping that copy is the difference between
        # finishing windowing in seconds and OOM-thrashing for minutes
        # on a 16 GB-RAM machine.
        windows = frames.reshape(n_full, self.L, self.F).astype(
            np.float32, copy=False,
        )
        label_blocks = frame_labels.reshape(n_full, self.L)

        strategy = self.WINDOW_LABEL_STRATEGY
        logger.info(
            "Windowing %d frames -> %d windows (strategy=%s)...",
            truncated, n_full, strategy,
        )

        # Vectorised label aggregation: build the (n_full, num_classes)
        # count matrix in one pass via per-class boolean sum-along-axis.
        # This is ~100x faster than the previous Python-loop +
        # per-window np.bincount approach for million-window datasets.
        counts = np.zeros((n_full, self.num_classes), dtype=np.int64)
        for c in range(self.num_classes):
            counts[:, c] = (label_blocks == c).sum(axis=1)

        if strategy == "majority":
            win_labels = counts.argmax(axis=1).astype(np.int64)
        elif strategy == "any-attack":
            # 0 if no attack frame in window, else argmax over [1..C).
            attack_counts = counts[:, 1:]
            has_attack = attack_counts.sum(axis=1) > 0
            attack_winner = attack_counts.argmax(axis=1) + 1
            win_labels = np.where(has_attack, attack_winner, 0).astype(np.int64)
        else:
            raise ValueError(
                f"Unknown WINDOW_LABEL_STRATEGY={strategy!r}; "
                f"expected 'majority' or 'any-attack'."
            )
        logger.info(
            "  windowed label dist = %s",
            np.bincount(win_labels, minlength=self.num_classes).tolist(),
        )
        return windows, win_labels

    def _sanity_check_windows(
        self, windows: np.ndarray, labels: np.ndarray
    ) -> None:
        if windows.ndim != 3 or windows.shape[1:] != (self.L, self.F):
            raise ValueError(
                f"prewindowed windows must be (N, {self.L}, {self.F}); "
                f"got {windows.shape}"
            )
        if labels.shape != (windows.shape[0],):
            raise ValueError(
                f"prewindowed labels {labels.shape} != windows[0] "
                f"{windows.shape[0]}"
            )
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                f"label values must be in [0, {self.num_classes}); "
                f"observed [{labels.min()}, {labels.max()}]"
            )

    # ------------------------------------------------------------------
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._windows.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._windows[idx], self._labels[idx]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def class_counts(self) -> np.ndarray:
        """Per-class window count for the *current split*."""
        return np.bincount(self._labels.numpy(), minlength=self.num_classes)
