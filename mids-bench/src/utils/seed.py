"""Global random seed management.

Every source of randomness in the framework should be routed through
:func:`set_global_seed` so that runs are bit-for-bit reproducible
(modulo non-deterministic CUDA kernels — see notes below).
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all visible CUDA devices).

    Args:
        seed: The seed value. The same value is fed to every RNG.
        deterministic: If True, also force cuDNN into deterministic mode
            and disable benchmarking. This makes runs bit-for-bit
            reproducible at the cost of speed; it also makes the
            selective-scan kernel in mamba-ssm raise on some PyTorch
            versions, so leave it False unless debugging numerics.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Default: prioritise throughput. cudnn.benchmark picks the best
        # conv algorithm per input shape on first call.
        torch.backends.cudnn.benchmark = True


def seeded_generator(seed: int) -> torch.Generator:
    """Return a CPU :class:`torch.Generator` seeded with ``seed``.

    Useful for DataLoader workers so that shuffling order is reproducible
    independently of the global RNG state.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
