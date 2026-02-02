"""Absoulte minimal experiment run using example configuration files."""

from pathlib import Path
EXAMPLE_DIR = Path("../../../examples/basic_usage/")
import sys
sys.path.insert(0, str(EXAMPLE_DIR))

import torch
from torch import nn
import torchmetrics

from venturi import Config, Experiment


def _get_simple_cnn(vcfg):
    return nn.Conv2d(1, 1, 3, padding=1)

def _get_simple_dataset(vcfg):
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, length):
            self.length = length
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return torch.randn(1, 28, 28), torch.randint(0, 2, (1, 28, 28)).float()
    
    return {
        "train_ds": SimpleDataset(16), "val_ds": SimpleDataset(16), "test_ds": SimpleDataset(16)}

def _get_metrics(vcfg):
    return torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="binary")
        }
    )

if __name__ == "__main__":  

    vcfg = Config(EXAMPLE_DIR / "config" / "base_config.yaml")
    vcfg.update_from_yaml("config/test_config.yaml")

    experiment = Experiment(vcfg)

    final_metric = experiment.fit()