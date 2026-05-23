"""Detector model registry.

To add a new baseline:
    1. Implement it as a :class:`BaseDetector` subclass under
       ``src/models/<name>.py``.
    2. Import it here and add it to ``MODEL_REGISTRY``.
    3. Drop a ``configs/model/<name>.yaml`` with its constructor kwargs.
    4. ``scripts/run_one.py --model <name>`` will pick it up.
"""

from .base import BaseDetector
from .canbus_ids import CanBusIDS
from .canshield import CANShield
from .cantransfer import CANTransfer
from .cantransformer import CanTransformer
from .cnn import CNN
from .dcnn import DCNN
from .gids import GIDS
from .mids import MIDS
from .mlp import MLP

MODEL_REGISTRY = {
    "mids": MIDS,
    "mlp": MLP,
    "cnn": CNN,
    "gids": GIDS,
    "dcnn": DCNN,
    "canbus_ids": CanBusIDS,
    "canshield": CANShield,
    "cantransfer": CANTransfer,
    "cantransformer": CanTransformer,
}

__all__ = [
    "BaseDetector",
    "MIDS",
    "MLP",
    "CNN",
    "GIDS",
    "DCNN",
    "CanBusIDS",
    "CANShield",
    "CANTransfer",
    "CanTransformer",
    "MODEL_REGISTRY",
]
