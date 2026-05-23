"""Dataset registry.

To add a new dataset:
    1. Implement it as a CANDataset subclass under src/datasets/<name>.py
    2. Register it in DATASET_REGISTRY below.
    3. Drop a configs/data/<name>.yaml describing where the data lives.
    4. scripts/run_one.py --dataset <name> will pick it up.
"""

from .base import CANDataset
from .crysys import CrysysDataset
from .ctnt import CtntDataset
from .otids import OtidsDataset
from .road import RoadDataset
from .tesla import TeslaDataset

DATASET_REGISTRY = {
    "tesla": TeslaDataset,
    "road": RoadDataset,
    "crysys": CrysysDataset,
    "otids": OtidsDataset,
    "ctnt": CtntDataset,
}

__all__ = [
    "CANDataset",
    "TeslaDataset",
    "RoadDataset",
    "CrysysDataset",
    "OtidsDataset",
    "CtntDataset",
    "DATASET_REGISTRY",
]
