"""Block-shuffled k-fold split protocol from §VI-A4 of the MIDS paper.

The procedure (recap):

1. Start with ``N`` non-overlapping windows of length ``L`` already
   carved out of the raw frame stream.
2. Re-group the windows into contiguous **chunks** of ``chunk_size``
   windows each (default 1000). Trailing windows that don't fill a
   chunk are dropped — this matches how the original MIDS code
   discarded incomplete blocks via ``reshape.py --discard_incomplete``.
3. Uniformly shuffle the chunks with a fixed seed (default 42).
4. Concatenate the shuffled chunks back into a single sequence and
   split it into ``n_folds`` contiguous blocks. Fold ``k`` uses block
   ``k`` as the test set; the rest are training.
5. Drop ``buffer_windows`` windows from the *training* side of every
   train/test seam to absorb any residual edge correlation
   (default ``buffer_windows = L`` per the paper).

The function returns numpy arrays of indices into the original
post-windowing tensor, ready to feed into ``torch.utils.data.Subset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class FoldIndices:
    """Per-fold index lists into the original window tensor."""

    train: np.ndarray  # 1-D int64
    test: np.ndarray  # 1-D int64


def build_block_shuffled_folds(
    n_windows: int,
    n_folds: int = 5,
    chunk_size: int = 1000,
    buffer_windows: int = 100,
    seed: int = 42,
) -> List[FoldIndices]:
    """Generate the per-fold train/test indices.

    Args:
        n_windows: Total number of non-overlapping windows.
        n_folds: Number of folds (default 5).
        chunk_size: Windows per chunk for the uniform shuffle stage
            (default 1000, matching the paper).
        buffer_windows: Number of training-side windows to drop on each
            side of every train/test seam (default 100 = L).
        seed: RNG seed for the chunk shuffle (default 42).

    Returns:
        A list of ``n_folds`` :class:`FoldIndices`, where ``train`` and
        ``test`` index into the original ``[0, n_windows)`` window space
        (i.e., they are *positions* in the un-shuffled tensor).

    Raises:
        ValueError: If there are too few windows to form one chunk per
            fold, or if the buffer would erase the entire training set.
    """
    if n_windows < chunk_size * n_folds:
        raise ValueError(
            f"Need at least chunk_size*n_folds = {chunk_size * n_folds} windows; "
            f"got {n_windows}. Reduce chunk_size or n_folds."
        )

    n_chunks = n_windows // chunk_size  # discard trailing partial chunk
    chunk_starts = np.arange(n_chunks, dtype=np.int64) * chunk_size

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_chunks)
    shuffled_chunk_starts = chunk_starts[perm]

    # Map shuffled position -> array of original-space window indices.
    # Each chunk contributes a contiguous block of length chunk_size.
    per_chunk_windows = np.stack(
        [np.arange(s, s + chunk_size, dtype=np.int64) for s in shuffled_chunk_starts],
        axis=0,
    )  # shape (n_chunks, chunk_size)

    # Split the shuffled chunk axis into n_folds contiguous blocks.
    fold_chunk_bounds = np.linspace(0, n_chunks, n_folds + 1, dtype=np.int64)

    folds: List[FoldIndices] = []
    for k in range(n_folds):
        lo, hi = fold_chunk_bounds[k], fold_chunk_bounds[k + 1]
        if hi - lo == 0:
            raise ValueError(
                f"Fold {k} ended up with zero chunks — increase n_windows or "
                f"decrease n_folds."
            )

        test_idx = per_chunk_windows[lo:hi].reshape(-1)

        train_chunks_left = per_chunk_windows[:lo]
        train_chunks_right = per_chunk_windows[hi:]

        # Apply the L-frame safety buffer on the training side of every
        # train/test seam. buffer_windows applies in *window* units; one
        # window already covers L frames, so a buffer of L windows is
        # extremely conservative — the paper says L *frames*, which in
        # window units is exactly 1. We expose buffer_windows so callers
        # can pick either interpretation.
        if buffer_windows > 0:
            if train_chunks_left.size > 0:
                train_left = train_chunks_left.reshape(-1)
                drop = min(buffer_windows, train_left.size)
                train_left = train_left[: train_left.size - drop]
            else:
                train_left = np.empty(0, dtype=np.int64)

            if train_chunks_right.size > 0:
                train_right = train_chunks_right.reshape(-1)
                drop = min(buffer_windows, train_right.size)
                train_right = train_right[drop:]
            else:
                train_right = np.empty(0, dtype=np.int64)
        else:
            train_left = train_chunks_left.reshape(-1)
            train_right = train_chunks_right.reshape(-1)

        train_idx = np.concatenate([train_left, train_right])

        if train_idx.size == 0:
            raise ValueError(
                f"Fold {k}: training set empty after applying buffer "
                f"({buffer_windows}). Reduce buffer_windows."
            )

        folds.append(FoldIndices(train=train_idx, test=test_idx))

    return folds


def save_folds(folds: List[FoldIndices], path: str) -> None:
    """Persist fold indices to an ``.npz`` so reruns reuse the split."""
    payload = {}
    for k, fi in enumerate(folds):
        payload[f"fold{k}_train"] = fi.train
        payload[f"fold{k}_test"] = fi.test
    np.savez(path, **payload)


def load_folds(path: str, n_folds: int) -> List[FoldIndices]:
    """Inverse of :func:`save_folds`."""
    data = np.load(path)
    return [
        FoldIndices(train=data[f"fold{k}_train"], test=data[f"fold{k}_test"])
        for k in range(n_folds)
    ]
