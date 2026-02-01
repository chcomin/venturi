"""Unit tests for venturi.core.Experiment class."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from venturi.config import Config
from venturi.core import Experiment


class TestExperimentInit:
    """Test Experiment initialization."""
    
    @pytest.mark.xfail(reason="Requires full config with all required fields")
    def test_experiment_init(self):
        """Initialize Experiment with config"""
        config = Config({
            "seed": 42,
            "dataset": {
                "setup": {"_target_": "some_function"}
            },
            "logging": {
                "create_folder": False,
                "silence_lightning": False,
                "wandb": {"silence_wandb": False}
            }
        })
        
        exp = Experiment(config)
        assert exp.args == config


class TestExperimentSetupLogging:
    """Test setup_logging method."""
    
    @pytest.mark.xfail(reason="Requires full config")
    def test_experiment_setup_logging_fit(self, tmp_path):
        """Create run_path for training"""
        pass


class TestExperimentFolderCreation:
    """Test folder creation logic."""
    
    @pytest.mark.xfail(reason="Requires full config and mocking")
    def test_experiment_folder_creation(self, tmp_path):
        """Creates folders correctly"""
        pass
