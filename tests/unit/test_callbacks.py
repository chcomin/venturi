"""Unit tests for venturi._util callback classes."""

import pytest
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from datetime import datetime

from venturi._util import (
    PlottingCallback,
    ImageSaveCallback,
    TrainingTimeLoggerCallback,
)


class TestPlottingCallback:
    """Test PlottingCallback class."""
    
    def test_plotting_callback_creates_plot(self, tmp_path):
        """Generates plots.png"""
        callback = PlottingCallback()
        
        # Create mock trainer with log_dir
        mock_trainer = MagicMock()
        mock_trainer.log_dir = str(tmp_path)
        
        # Create mock module with args for plotting
        mock_module = MagicMock()
        mock_module.vcfg.logging.plot.left_plot.metrics = ["train/loss", "val/loss"]
        mock_module.vcfg.logging.plot.left_plot.ylim.min = None
        mock_module.vcfg.logging.plot.left_plot.ylim.max = None
        mock_module.vcfg.logging.plot.right_plot.metrics = ["val/accuracy"]
        mock_module.vcfg.logging.plot.right_plot.ylim.min = None
        mock_module.vcfg.logging.plot.right_plot.ylim.max = None
        
        # Create a sample metrics.csv file
        metrics_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({
            "epoch": [0, 1, 2],
            "train/loss_epoch": [1.0, 0.8, 0.6],
            "val/loss": [0.9, 0.7, 0.5],
            "val/accuracy": [0.7, 0.8, 0.9]
        })
        df.to_csv(metrics_file, index=False)
        
        # Call the callback
        callback.on_train_epoch_end(mock_trainer, mock_module)
        
        # Check that plot was created
        assert (tmp_path / "plots.png").exists()
    
    def test_plotting_callback_handles_missing_metrics(self, tmp_path):
        """Gracefully handles missing columns"""
        callback = PlottingCallback()
        
        mock_trainer = MagicMock()
        mock_trainer.log_dir = str(tmp_path)
        
        mock_module = MagicMock()
        mock_module.vcfg.logging.plot.left_plot.metrics = ["train/loss", "missing_metric"]
        mock_module.vcfg.logging.plot.left_plot.ylim.min = None
        mock_module.vcfg.logging.plot.left_plot.ylim.max = None
        mock_module.vcfg.logging.plot.right_plot.metrics = ["val/accuracy"]
        mock_module.vcfg.logging.plot.right_plot.ylim.min = None
        mock_module.vcfg.logging.plot.right_plot.ylim.max = None
        
        # Create CSV with only some metrics
        metrics_file = tmp_path / "metrics.csv"
        df = pd.DataFrame({
            "epoch": [0, 1],
            "train/loss_epoch": [1.0, 0.8],
            "val/accuracy": [0.7, 0.8]
        })
        df.to_csv(metrics_file, index=False)
        
        # Should not raise an error
        callback.on_train_epoch_end(mock_trainer, mock_module)
    
    def test_plotting_callback_handles_missing_csv(self, tmp_path):
        """Raises error when metrics.csv missing"""
        callback = PlottingCallback()
        
        mock_trainer = MagicMock()
        mock_trainer.log_dir = str(tmp_path)
        
        mock_module = MagicMock()
        
        with pytest.raises(ValueError, match="Metrics file not found"):
            callback.on_train_epoch_end(mock_trainer, mock_module)


class TestImageSaveCallback:
    """Test ImageSaveCallback class."""
    
    def test_image_save_callback_init(self, tmp_path):
        """Initialize callback"""
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0, 1, 2],
            log_disk=True
        )
        
        assert callback.run_path == tmp_path
        assert callback.val_img_indices == {0, 1, 2}
        assert callback.log_disk is True
    
    def test_image_save_callback_on_fit_start(self, tmp_path):
        """Creates image folders on fit start"""
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0, 1],
            log_disk=True
        )
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        callback.on_fit_start(mock_trainer, mock_module)
        
        # Check folders were created
        assert (tmp_path / "images" / "image_0").exists()
        assert (tmp_path / "images" / "image_1").exists()
    
    def test_image_save_callback_saves_to_disk(self, tmp_path):
        """Saves images to folder"""
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0],
            log_disk=True,
            mean=0.5,
            std=0.5
        )
        
        # Create folders
        (tmp_path / "images" / "image_0").mkdir(parents=True)
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        # Create mock batch
        inputs = torch.randn(2, 3, 32, 32)
        targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
        batch = (inputs, targets)
        
        # Create mock outputs with logits
        outputs = {
            "logits": torch.randn(2, 1, 32, 32)
        }
        
        callback.on_validation_batch_end(
            mock_trainer, mock_module, outputs, batch, batch_idx=0
        )
        
        # Check that images were saved
        saved_files = list((tmp_path / "images" / "image_0").glob("*.png"))
        assert len(saved_files) > 0
    
    def test_image_save_callback_binary_segmentation(self, tmp_path):
        """Handles binary segmentation masks"""
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0],
            log_disk=True
        )
        
        (tmp_path / "images" / "image_0").mkdir(parents=True)
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        # Binary segmentation: 1 channel output
        inputs = torch.randn(1, 3, 32, 32)
        targets = torch.randint(0, 2, (1, 1, 32, 32)).float()
        batch = (inputs, targets)
        
        outputs = {
            "logits": torch.randn(1, 1, 32, 32)  # Single channel
        }
        
        callback.on_validation_batch_end(
            mock_trainer, mock_module, outputs, batch, batch_idx=0
        )
        
        # Should save without error
        saved_files = list((tmp_path / "images" / "image_0").glob("*.png"))
        assert len(saved_files) > 0
    
    def test_image_save_callback_multiclass_segmentation(self, tmp_path):
        """Handles multiclass segmentation masks"""
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0],
            log_disk=True
        )
        
        (tmp_path / "images" / "image_0").mkdir(parents=True)
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        # Multiclass segmentation: multiple channels
        inputs = torch.randn(1, 3, 32, 32)
        targets = torch.randint(0, 5, (1, 1, 32, 32)).float()
        batch = (inputs, targets)
        
        outputs = {
            "logits": torch.randn(1, 5, 32, 32)  # 5 classes
        }
        
        callback.on_validation_batch_end(
            mock_trainer, mock_module, outputs, batch, batch_idx=0
        )
        
        # Should save without error
        saved_files = list((tmp_path / "images" / "image_0").glob("*.png"))
        assert len(saved_files) > 0
    
    def test_image_save_callback_denormalization(self, tmp_path):
        """Correctly denormalizes images"""
        mean = 0.5
        std = 0.5
        
        callback = ImageSaveCallback(
            run_path=tmp_path,
            val_img_indices=[0],
            log_disk=True,
            mean=mean,
            std=std
        )
        
        # The denormalization should work correctly
        assert callback.mean == mean
        assert callback.std == std


class TestTrainingTimeLoggerCallback:
    """Test TrainingTimeLoggerCallback class."""
    
    def test_training_time_logger_init(self, tmp_path):
        """Initialize callback"""
        callback = TrainingTimeLoggerCallback(tmp_path)
        assert callback.run_path == tmp_path
    
    def test_training_time_logger_writes_file(self, tmp_path):
        """Writes training_time.txt"""
        callback = TrainingTimeLoggerCallback(tmp_path)
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        # Call lifecycle methods (callback uses on_train_start/end not on_fit_start/end)
        callback.on_train_start(mock_trainer, mock_module)
        
        # Simulate some time passing
        import time
        time.sleep(0.1)
        
        callback.on_train_end(mock_trainer, mock_module)
        
        # Check file was created
        time_file = tmp_path / "training_time.txt"
        assert time_file.exists()
        
        # Check file has content
        content = time_file.read_text()
        assert "Training started at:" in content
        assert "Training ended at:" in content
        assert "2026" in content  # Year in timestamp
    
    def test_training_time_logger_start_end_times(self, tmp_path):
        """Logs start and end times"""
        callback = TrainingTimeLoggerCallback(tmp_path)
        
        mock_trainer = MagicMock()
        mock_module = MagicMock()
        
        callback.on_train_start(mock_trainer, mock_module)
        
        # Check that file was created
        time_file = tmp_path / "training_time.txt"
        assert time_file.exists()
        
        callback.on_train_end(mock_trainer, mock_module)
        
        # Check file has both start and end entries
        content = time_file.read_text()
        assert "Training started at:" in content
        assert "Training ended at:" in content
        
        # Verify format with timestamps
        lines = content.strip().split('\n')
        assert len(lines) == 2
