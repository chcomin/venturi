"""Unit tests for venturi.core.DataModule class."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.utils.data as data

from venturi.config import Config
from venturi.core import DataModule


class MockDataset(data.Dataset):
    """Mock dataset for testing."""
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0, 10, (1,))


def mock_dataset_setup_fit(args):
    """Mock dataset setup function for fit stage."""
    train_ds = MockDataset(100)
    val_ds = MockDataset(20)
    return train_ds, val_ds


def mock_dataset_setup_test(args):
    """Mock dataset setup function for test stage."""
    return MockDataset(50)


def mock_dataloader(dataset, batch_size=4, shuffle=False, generator=None):
    """Mock dataloader factory."""
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


class TestDataModuleInit:
    """Test DataModule initialization."""
    
    def test_datamodule_init(self):
        """Initialize with valid config"""
        config = Config({
            "seed": 42,
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True}
            }
        })
        
        dm = DataModule(config)
        assert dm.args == config
        assert dm.ds_dict == {}


class TestDataModuleSetup:
    """Test DataModule setup method."""
    
    def test_datamodule_setup_fit(self, monkeypatch):
        """Call setup with stage='fit'"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                "train_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                },
                "val_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="fit")
        
        assert hasattr(dm, "train_ds")
        assert hasattr(dm, "val_ds")
        assert isinstance(dm.train_ds, MockDataset)
        assert isinstance(dm.val_ds, MockDataset)
        assert hasattr(dm, "generator")
    
    def test_datamodule_setup_test(self):
        """Call setup with stage='test'"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_test", "_partial_": True},
                "test_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="test")
        
        assert hasattr(dm, "test_ds")
        assert isinstance(dm.test_ds, MockDataset)
    
    def test_datamodule_setup_none_stage(self):
        """Call setup with stage=None (should behave like fit)"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                "train_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                },
                "val_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage=None)
        
        assert hasattr(dm, "train_ds")
        assert hasattr(dm, "val_ds")


class TestDataModuleDataloaders:
    """Test DataModule dataloader methods."""
    
    def test_datamodule_train_dataloader(self):
        """Returns correct DataLoader"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                "train_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 8,
                    "shuffle": True
                },
                "val_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 8
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="fit")
        
        train_dl = dm.train_dataloader()
        assert isinstance(train_dl, data.DataLoader)
        assert train_dl.batch_size == 8
    
    def test_datamodule_val_dataloader(self):
        """Returns validation DataLoader"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                "train_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                },
                "val_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 16
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="fit")
        
        val_dl = dm.val_dataloader()
        assert isinstance(val_dl, data.DataLoader)
        assert val_dl.batch_size == 16
    
    def test_datamodule_test_dataloader(self):
        """Returns test DataLoader"""
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_test", "_partial_": True},
                "test_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 32
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="test")
        
        test_dl = dm.test_dataloader()
        assert isinstance(test_dl, data.DataLoader)
        assert test_dl.batch_size == 32


class TestDataModuleGenerator:
    """Test generator seeding."""
    
    def test_datamodule_generator_seed(self):
        """Generator uses correct seed"""
        config = Config({
            "seed": 123,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                "train_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                },
                "val_dataloader": {
                    "_target_": "mock_dataloader",
                    "_partial_": True,
                    "batch_size": 4
                }
            }
        })
        
        dm = DataModule(config)
        dm.setup(stage="fit")
        
        # Check generator was created
        assert hasattr(dm, "generator")
        assert isinstance(dm.generator, torch.Generator)


class TestDataModuleErrorHandling:
    """Test error handling."""
    
    def test_datamodule_setup_wrong_dataset_count_fit(self):
        """Raise error if setup function returns wrong number of datasets for fit"""
        def bad_setup(args):
            return MockDataset(100)  # Returns 1 dataset instead of 2
        
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "bad_setup", "_partial_": True}
            }
        })
        
        dm = DataModule(config)
        
        with pytest.raises(ValueError, match="must return a train and validation dataset"):
            dm.setup(stage="fit")
    
    def test_datamodule_setup_wrong_type_test(self):
        """Raise error if setup function returns wrong type for test"""
        def bad_setup(args):
            return [MockDataset(50), MockDataset(50)]  # Returns list instead of single dataset
        
        config = Config({
            "seed": 42,
            "logging": {
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            },
            "dataset": {
                "setup": {"_target_": "bad_setup", "_partial_": True}
            }
        })
        
        dm = DataModule(config)
        
        with pytest.raises(ValueError, match="must return a Pytorch Dataset"):
            dm.setup(stage="test")


class TestDataModuleSilencing:
    """Test silencing of lightning and wandb."""
    
    def test_datamodule_silence_lightning(self, monkeypatch):
        """Test silencing in multiprocessing"""
        silence_called = []
        
        def mock_silence():
            silence_called.append(True)
        
        with patch("venturi.core.silence_lightning", mock_silence):
            config = Config({
                "seed": 42,
                "logging": {
                    "silence_lightning": True,
                    "wandb": {"silence_wandb": False}
                },
                "dataset": {
                    "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                    "train_dataloader": {
                        "_target_": "mock_dataloader",
                        "_partial_": True,
                        "batch_size": 4
                    },
                    "val_dataloader": {
                        "_target_": "mock_dataloader",
                        "_partial_": True,
                        "batch_size": 4
                    }
                }
            })
            
            dm = DataModule(config)
            dm.setup(stage="fit")
            
            assert len(silence_called) == 1
    
    def test_datamodule_silence_wandb(self, monkeypatch):
        """Test wandb silencing via environment variable"""
        with patch("venturi.core._has_wandb", True):
            config = Config({
                "seed": 42,
                "logging": {
                    "silence_lightning": False,
                    "wandb": {"silence_wandb": True}
                },
                "dataset": {
                    "setup": {"_target_": "mock_dataset_setup_fit", "_partial_": True},
                    "train_dataloader": {
                        "_target_": "mock_dataloader",
                        "_partial_": True,
                        "batch_size": 4
                    },
                    "val_dataloader": {
                        "_target_": "mock_dataloader",
                        "_partial_": True,
                        "batch_size": 4
                    }
                }
            })
            
            dm = DataModule(config)
            dm.setup(stage="fit")
            
            assert os.environ.get("WANDB_SILENT") == "True"
