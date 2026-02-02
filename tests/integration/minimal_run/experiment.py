"""Absoulte minimal experiment run using example configuration files."""

import sys
from pathlib import Path

import torch
import torchmetrics
from torch import nn

from venturi import Config, Experiment

EXAMPLE_DIR = Path("../../../examples/basic_usage/")

sys.path.insert(0, str(EXAMPLE_DIR))



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