from typing import Any

from venturi.config import Config, instantiate
from venturi.core import DataModule, Experiment, TrainingModule
from venturi.util import MetricCollectionWrapper

__version__ = "0.9.3"

__all__ = [
    "Config",
    "DataModule",
    "Experiment",
    "MetricCollectionWrapper",
    "TrainingModule",
    "__version__",
    "instantiate",
]

type VenturiConfig = Any
