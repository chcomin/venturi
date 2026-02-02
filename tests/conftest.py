import pytest
import torch
import torch.nn as nn
import torchmetrics

from venturi.config import Config


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


def _get_simple_cnn(vcfg):
    return nn.Conv2d(1, 1, 3, padding=1)

def _get_metrics(vcfg):
    return torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="binary")
        }
    )


@pytest.fixture
def base_vcfg():
    """Returns a minimal valid Config object."""

    custom_vcfg = {
        "logging": {
            "create_folder": False,
            "log_csv": False,
            "log_plot": False,
            "log_training_time": False,
        },
        "dataset": {
            "setup": {
                "_target_": "tests.conftest._get_simple_dataset"
            }
        },
        "model": {
            "setup": {
                "_target_": "tests.conftest._get_simple_cnn"
            }
            },
        "losses": {
            "bce": {
                "instance": {
                    "_target_": "torch.nn.BCEWithLogitsLoss"
                }
            }
        },
        "metrics": {
            "setup": {
                "_target_": "tests.conftest._get_metrics"
            }
        }
    }

    vcfg = Config("venturi/base_config.yaml")
    vcfg.update_from_dict(custom_vcfg)
    return vcfg


@pytest.fixture
def simple_cls_model():
    return nn.Linear(10, 2)


@pytest.fixture
def simple_seg_model():
    return nn.Conv2d(1, 1, 3, padding=1)
