"""Unit tests for venturi._util callback classes."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from venturi._util import (
    PlottingCallback,
    ImageSaveCallback,
    TrainingTimeLoggerCallback,
)


class TestPlottingCallback:
    """Test PlottingCallback class."""
    
    @pytest.mark.xfail(reason="Requires mocking trainer and CSV file")
    def test_plotting_callback_creates_plot(self, tmp_path):
        """Generates plots.png"""
        pass
    
    @pytest.mark.xfail(reason="Requires complex setup")
    def test_plotting_callback_handles_missing_metrics(self):
        """Gracefully handles missing columns"""
        pass


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
    
    @pytest.mark.xfail(reason="Requires mocking trainer and module")
    def test_image_save_callback_saves_to_disk(self, tmp_path):
        """Saves images to folder"""
        pass


class TestTrainingTimeLoggerCallback:
    """Test TrainingTimeLoggerCallback class."""
    
    def test_training_time_logger_init(self, tmp_path):
        """Initialize callback"""
        callback = TrainingTimeLoggerCallback(tmp_path)
        assert callback.run_path == tmp_path
    
    @pytest.mark.xfail(reason="Requires mocking trainer")
    def test_training_time_logger_writes_file(self, tmp_path):
        """Writes training_time.txt"""
        pass
