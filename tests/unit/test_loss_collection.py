"""Unit tests for venturi._util.LossCollection class."""

import torch
import torch.nn as nn

from venturi.util import LossCollection
from venturi.config import Config


class TestLossCollectionInit:
    """Test LossCollection initialization."""
    
    def test_loss_collection_single_loss(self):
        """Single loss with weight"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        loss_collection = LossCollection(config)
        assert "mse" in loss_collection.loss_map
        assert loss_collection.weights["mse"] == 1.0
    
    def test_loss_collection_multiple_losses(self):
        """Combine multiple losses"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            },
            "l1": {
                "instance": {"_target_": "torch.nn.L1Loss"},
                "loss_weight": 0.5
            }
        })
        
        loss_collection = LossCollection(config)
        assert "mse" in loss_collection.loss_map
        assert "l1" in loss_collection.loss_map
        assert loss_collection.weights["mse"] == 1.0
        assert loss_collection.weights["l1"] == 0.5
    
    def test_loss_collection_single_loss_no_weight(self):
        """Auto-assign weight=1.0"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"}
            }
        })
        
        loss_collection = LossCollection(config)
        assert loss_collection.weights["mse"] == 1.0


class TestLossCollectionForward:
    """Test LossCollection forward pass."""
    
    def test_loss_collection_forward(self):
        """Compute weighted sum"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        loss_collection = LossCollection(config)
        
        input_tensor = torch.randn(4, 10)
        target_tensor = torch.randn(4, 10)
        
        total_loss, logs = loss_collection(input_tensor, target_tensor)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert "mse" in logs
        assert isinstance(logs["mse"], torch.Tensor)
    
    def test_loss_collection_return_logs(self):
        """Return individual losses"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        loss_collection = LossCollection(config, return_logs=True)
        
        input_tensor = torch.randn(4, 10)
        target_tensor = torch.randn(4, 10)
        
        total_loss, logs = loss_collection(input_tensor, target_tensor)
        
        assert isinstance(logs, dict)
        assert "mse" in logs
    
    def test_loss_collection_no_logs(self):
        """Don't return logs when return_logs=False"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        loss_collection = LossCollection(config, return_logs=False)
        
        input_tensor = torch.randn(4, 10)
        target_tensor = torch.randn(4, 10)
        
        result = loss_collection(input_tensor, target_tensor)
        
        # Should just be a tensor, not a tuple
        assert isinstance(result, torch.Tensor)
    
    def test_loss_collection_weighted_sum(self):
        """Verify weighted sum calculation"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 2.0
            },
            "l1": {
                "instance": {"_target_": "torch.nn.L1Loss"},
                "loss_weight": 0.5
            }
        })
        
        loss_collection = LossCollection(config)
        
        input_tensor = torch.randn(4, 10)
        target_tensor = torch.randn(4, 10)
        
        total_loss, logs = loss_collection(input_tensor, target_tensor)
        
        # Manually compute expected loss
        mse_fn = nn.MSELoss()
        l1_fn = nn.L1Loss()
        expected = 2.0 * mse_fn(input_tensor, target_tensor) + 0.5 * l1_fn(input_tensor, target_tensor)
        
        assert torch.allclose(total_loss, expected, atol=1e-6)


class TestLossCollectionClone:
    """Test LossCollection clone method."""
    
    def test_loss_collection_clone(self):
        """Clone with prefix"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        original = LossCollection(config)
        cloned = original.clone(prefix="train/")
        
        assert "train/mse" in cloned.loss_map
        assert cloned.weights["train/mse"] == 1.0
        assert original.loss_map is not cloned.loss_map
    
    def test_loss_collection_clone_no_prefix(self):
        """Clone without prefix"""
        config = Config({
            "mse": {
                "instance": {"_target_": "torch.nn.MSELoss"},
                "loss_weight": 1.0
            }
        })
        
        original = LossCollection(config)
        cloned = original.clone()
        
        assert "mse" in cloned.loss_map
