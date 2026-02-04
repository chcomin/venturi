import sys
from pathlib import Path

import torch

from venturi import Config, Experiment, TrainingModule

EXAMPLE_DIR = Path("../../../examples/hyperparameter_search/")
sys.path.insert(0, str(EXAMPLE_DIR))

from experiment import MockExperiment

if __name__ == "__main__":
    
    vcfg = Config(EXAMPLE_DIR / "config" / "base_config.yaml")
    vcfg.update_from_yaml(EXAMPLE_DIR / "config" / "exp_config.yaml")
    experiment = MockExperiment(vcfg)

    #experiment.fit()

    vcfg_space = Config(EXAMPLE_DIR / "config" / "search_space.yaml")
    study = experiment.optimize(vcfg_space)

    print("Best parameters:", study.best_params)