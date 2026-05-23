"""Stdlib :mod:`logging` setup with simultaneous console + file output."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "mids_bench",
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Build a logger that writes to stdout and (optionally) a file.

    Subsequent calls with the same ``name`` reuse the existing logger
    rather than stacking handlers, so this is safe to call from multiple
    entry points.

    Args:
        name: Logger name. Use a per-component name (``mids_bench.trainer``)
            to leverage the standard logging hierarchy.
        log_file: If given, append-mode file handler is added at this path.
            Parent directories are created on demand.
        level: Logging level for both handlers.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers when called twice.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
