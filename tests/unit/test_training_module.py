"""Unit tests for venturi.core.TrainingModule class."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from venturi.config import Config
from venturi.core import TrainingModule


def mock_model_setup(vcfg):
    """Mock model setup function."""
    return nn.Linear(10, 2)


def mock_metrics_setup(vcfg):
    """Mock metrics setup function."""
    class MockMetrics(nn.Module):
        def forward(self, logits, targets):
            return {"accuracy": torch.tensor(0.95)}
        
        def clone(self, prefix=""):
            cloned = MockMetrics()
            cloned.prefix = prefix
            return cloned
    
    return MockMetrics()


class TestTrainingModuleInit:
    """Test TrainingModule initialization."""
    
    def test_training_module_init(self, base_vcfg: Config):
        """Initialize with valid config"""
        module = TrainingModule(base_vcfg)
        
        assert module.vcfg == base_vcfg
        assert hasattr(module, "model")
        assert hasattr(module, "train_loss")
        assert hasattr(module, "val_loss")
        assert hasattr(module, "val_metrics")
        assert hasattr(module, "test_metrics")
    
    def test_training_module_init_with_mock_config(self):
        """Initialize with mock config"""
        config = Config({
            "model": {
                "setup": {"_target_": "mock_model_setup", "_partial_": True}
            },
            "losses": {
                "ce": {
                    "instance": {"_target_": "torch.nn.CrossEntropyLoss"},
                    "loss_weight": 1.0
                }
            },
            "metrics": {
                "setup": {"_target_": "mock_metrics_setup", "_partial_": True}
            }
        })
        
        module = TrainingModule(config)
        assert module.vcfg == config
        assert hasattr(module, "model")


class TestTrainingModuleForward:
    """Test TrainingModule forward pass."""
    
    def test_training_module_forward(self, base_vcfg: Config):
        """Forward pass works"""
        module = TrainingModule(base_vcfg)
        
        # Use the correct input shape for the CNN model (1, 28, 28)
        x = torch.randn(4, 1, 28, 28)
        output = module(x)
        
        # Output should have same spatial dimensions with 1 channel
        assert output.shape == (4, 1, 28, 28)
    
    def test_training_module_forward_with_mock_model(self):
        """Forward pass with mock linear model"""
        config = Config({
            "model": {
                "setup": {"_target_": "mock_model_setup", "_partial_": True}
            },
            "losses": {
                "ce": {
                    "instance": {"_target_": "torch.nn.CrossEntropyLoss"},
                    "loss_weight": 1.0
                }
            },
            "metrics": {
                "setup": {"_target_": "mock_metrics_setup", "_partial_": True}
            }
        })
        
        module = TrainingModule(config)
        x = torch.randn(4, 10)
        output = module(x)
        assert output.shape == (4, 2)


class TestTrainingModuleSteps:
    """Test training, validation, and test steps."""
    
    def test_training_module_training_step(self, base_vcfg: Config):
        """Training step logic"""
        module = TrainingModule(base_vcfg)
        
        # Create synthetic batch matching SimpleDataset format
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 2, (4, 1, 28, 28)).float()
        batch = (x, y)
        
        # Mock trainer
        module.trainer = MagicMock()
        module.trainer.global_step = 0
        module.log = MagicMock()  # Mock the log method
        
        loss = module.training_step(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.requires_grad
    
    def test_training_module_validation_step(self, base_vcfg: Config):
        """Validation step logic"""
        module = TrainingModule(base_vcfg)
        
        # Create synthetic batch
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 2, (4, 1, 28, 28)).float()
        batch = (x, y)
        
        # Mock logging methods
        module.log = MagicMock()
        module.log_dict = MagicMock()
        
        result = module.validation_step(batch)
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert "logits" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert isinstance(result["logits"], torch.Tensor)
    
    def test_training_module_test_step(self, base_vcfg: Config):
        """Test step logic"""
        module = TrainingModule(base_vcfg)
        
        # Create synthetic batch
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 2, (4, 1, 28, 28)).float()
        batch = (x, y)
        
        # Mock logging method
        module.log_dict = MagicMock()
        
        # Test step doesn't return anything, just logs
        result = module.test_step(batch)
        
        # Should not raise an error
        assert result is None or isinstance(result, dict)
    
    def test_training_module_predict_step(self, base_vcfg: Config):
        """Predict step logic"""
        module = TrainingModule(base_vcfg)
        
        # Create synthetic batch (just input, no target)
        x = torch.randn(4, 1, 28, 28)
        
        output = module.predict_step(x)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (4, 1, 28, 28)


class TestTrainingModuleOptimizer:
    """Test optimizer configuration."""
    
    def test_training_module_configure_optimizers(self, base_vcfg: Config):
        """Returns optimizer dict"""
        base_vcfg.training.optimizer = {
            "_target_": "torch.optim.SGD",
            "_partial_": True,
            "lr": 0.01
        }
        
        module = TrainingModule(base_vcfg)
        # Mock trainer (needed for logging)
        module.trainer = MagicMock()
        result = module.configure_optimizers()
        
        assert "optimizer" in result
        assert isinstance(result["optimizer"], torch.optim.SGD)
    
    def test_training_module_configure_optimizers_with_scheduler(self, base_vcfg: Config):
        """Returns optimizer and scheduler"""
        base_vcfg.training.optimizer = {
            "_target_": "torch.optim.SGD",
            "_partial_": True,
            "lr": 0.01
        }
        base_vcfg.training.lr_scheduler = Config({
            "instance": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "_partial_": True,
                "step_size": 10,
                "gamma": 0.1
            },
            "scheduler_config": {
                "interval": "epoch"
            }
        })
        
        module = TrainingModule(base_vcfg)
        
        # Mock trainer for scheduler that needs total steps
        module.trainer = MagicMock()
        module.trainer.max_epochs = 10
        
        result = module.configure_optimizers()
        
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], torch.optim.SGD)


class TestTrainingModuleScheduler:
    """Test learning rate scheduler configuration."""
    
    def test_training_module_get_scheduler(self, base_vcfg: Config):
        """Get scheduler config"""
        base_vcfg.training.lr_scheduler = Config({
            "instance": {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "_partial_": True,
                "step_size": 10,
                "gamma": 0.1
            },
            "scheduler_config": {
                "interval": "epoch"
            }
        })
        
        module = TrainingModule(base_vcfg)
        module.trainer = MagicMock()
        module.trainer.max_epochs = 10
        
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        scheduler_config = module.get_scheduler(optimizer)
        
        assert "scheduler" in scheduler_config
        assert "interval" in scheduler_config
        assert scheduler_config["interval"] == "epoch"
    
    def test_training_module_estimate_total_steps(self, base_vcfg: Config):
        """Estimate total training steps"""
        module = TrainingModule(base_vcfg)
        
        # Mock trainer with necessary attributes
        module.trainer = MagicMock()
        module.trainer.max_epochs = 10
        module.trainer.estimated_stepping_batches = 100
        
        total_steps = module._estimate_total_steps()
        
        assert total_steps == 100
    
    def test_training_module_scheduler_needs_total_iters(self, base_vcfg: Config):
        """Scheduler that needs total_iters"""
        base_vcfg.training.lr_scheduler = Config({
            "instance": {
                "_target_": "torch.optim.lr_scheduler.LinearLR",
                "_partial_": True,
            },
            "scheduler_config": {
                "interval": "step"
            },
            "needs_total_iters": True
        })
        
        module = TrainingModule(base_vcfg)
        
        # Mock trainer
        module.trainer = MagicMock()
        module.trainer.max_epochs = 5
        module.trainer.estimated_stepping_batches = 50
        
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
        scheduler_config = module.get_scheduler(optimizer)
        
        assert "scheduler" in scheduler_config
        # The scheduler should have been created with total_iters


class TestTrainingModuleErrorHandling:
    """Test error handling."""
    
    def test_training_module_estimate_total_steps_infinite_epochs(self, base_vcfg: Config):
        """Raise error when max_epochs=-1 and scheduler needs total steps"""
        module = TrainingModule(base_vcfg)
        
        # Mock trainer with infinite epochs
        module.trainer = MagicMock()
        module.trainer.max_epochs = -1
        module.trainer.estimated_stepping_batches = float("inf")
        
        with pytest.raises(ValueError, match="max_epochs.*is set to -1"):
            module._estimate_total_steps()
