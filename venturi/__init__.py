from typing import Any

from venturi.config import Config, instantiate
from venturi.core import DataModule, Experiment, TrainingModule

__version__ = "0.9.1"

__all__ = [
    "Config",
    "DataModule",
    "Experiment",
    "TrainingModule",
    "__version__",
    "instantiate",
]

type VenturiConfig = Any
