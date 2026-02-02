"""Unit tests for venturi.core.Experiment class."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch

# Import the helper functions from conftest to make them available in scope
from venturi.config import Config
from venturi.core import DataModule, Experiment, TrainingModule


class TestExperimentInit:
    """Test Experiment initialization."""
    
    def test_experiment_init(self, base_vcfg: Config):
        """Initialize Experiment with config"""
        exp = Experiment(base_vcfg)
        assert exp.vcfg == base_vcfg
        assert exp.data_module is not None
        assert isinstance(exp.data_module, DataModule)
        assert exp.run_path is None
        assert exp.model is None
        assert exp.trainer is None
    
    def test_experiment_init_sets_seed(self, base_vcfg: Config):
        """Initialization sets random seed"""
        seed_value = 12345
        base_vcfg.seed = seed_value
        
        with patch("pytorch_lightning.seed_everything") as mock_seed:
            exp = Experiment(base_vcfg)
            mock_seed.assert_called()


class TestExperimentSetupLogging:
    """Test setup_logging method."""
    
    def test_experiment_setup_logging_fit(self, base_vcfg: Config, tmp_path):
        """Create run_path for training"""
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        
        exp = Experiment(base_vcfg)
        run_path = exp.setup_logging(stage="fit")
        
        assert run_path == tmp_path / "test_run"
        assert run_path.exists()
        assert (run_path / "config.yaml").exists()
    
    def test_experiment_setup_logging_test(self, base_vcfg: Config, tmp_path):
        """Run_path for testing doesn't create folders"""
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        
        exp = Experiment(base_vcfg)
        run_path = exp.setup_logging(stage="test")
        
        assert run_path == tmp_path / "test_run"
        # Folder should not be created for test stage
        assert not run_path.exists()
    
    def test_experiment_setup_logging_without_folder_creation(self, base_vcfg: Config, tmp_path):
        """No folder creation when create_folder=False"""
        base_vcfg.logging.create_folder = False
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        
        exp = Experiment(base_vcfg)
        run_path = exp.setup_logging(stage="fit")
        
        assert run_path == tmp_path / "test_run"
        assert not run_path.exists()


class TestExperimentFolderCreation:
    """Test folder creation logic."""
    
    def test_experiment_folder_creation(self, base_vcfg: Config, tmp_path):
        """Creates folders correctly"""
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "experiment")
        
        exp = Experiment(base_vcfg)
        run_path = exp.setup_logging(stage="fit")
        
        assert run_path.exists()
        assert run_path.is_dir()
    
    def test_experiment_overwrite_existing(self, base_vcfg: Config, tmp_path):
        """Overwrites existing run_path when overwrite_existing=True"""
        run_path = tmp_path / "existing_run"
        run_path.mkdir()
        (run_path / "old_file.txt").write_text("old content")
        
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.overwrite_existing = True
        base_vcfg.logging.run_path = str(run_path)
        
        exp = Experiment(base_vcfg)
        exp.setup_logging(stage="fit")
        
        # Old file should be gone
        assert not (run_path / "old_file.txt").exists()
        # New config should exist
        assert (run_path / "config.yaml").exists()
    
    def test_experiment_get_next_name(self, base_vcfg: Config, tmp_path):
        """Increments run_path name when overwrite_existing=False"""
        run_path = tmp_path / "run"
        run_path.mkdir()
        
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.overwrite_existing = False
        base_vcfg.logging.run_path = str(run_path)
        
        exp = Experiment(base_vcfg)
        new_run_path = exp.setup_logging(stage="fit")
        
        assert new_run_path == tmp_path / "run_2"
        assert new_run_path.exists()


class TestExperimentDataModule:
    """Test get_data_module method."""
    
    def test_experiment_get_data_module(self, base_vcfg: Config):
        """Returns DataModule instance"""
        exp = Experiment(base_vcfg)
        dm = exp.get_data_module()
        
        assert isinstance(dm, DataModule)
        assert dm.vcfg == base_vcfg


class TestExperimentModel:
    """Test get_model method."""
    
    def test_experiment_get_model(self, base_vcfg: Config):
        """Returns TrainingModule instance"""
        exp = Experiment(base_vcfg)
        model = exp.get_model()
        
        assert isinstance(model, TrainingModule)
        assert model.vcfg == base_vcfg


class TestExperimentLoggers:
    """Test get_loggers method."""
    
    def test_experiment_get_loggers_csv(self, base_vcfg: Config, tmp_path):
        """Setup CSV logger"""
        base_vcfg.logging.log_csv = True
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "run")
        
        exp = Experiment(base_vcfg)
        exp.run_path = tmp_path / "run"
        
        loggers = exp.get_loggers()
        
        assert len(loggers) >= 1
        assert any("CSV" in str(type(logger).__name__) for logger in loggers)
    
    def test_experiment_get_loggers_requires_run_path(self, base_vcfg: Config):
        """Raises error if run_path not set"""
        exp = Experiment(base_vcfg)
        exp.run_path = None
        
        with pytest.raises(ValueError, match="run_path must be set"):
            exp.get_loggers()


class TestExperimentCallbacks:
    """Test get_callbacks method."""
    
    def test_experiment_get_callbacks_requires_run_path(self, base_vcfg: Config):
        """Raises error if run_path not set"""
        exp = Experiment(base_vcfg)
        exp.run_path = None
        
        with pytest.raises(ValueError, match="run_path must be set"):
            exp.get_callbacks()
    
    def test_experiment_get_callbacks_basic(self, base_vcfg: Config, tmp_path):
        """Returns list of callbacks"""
        exp = Experiment(base_vcfg)
        exp.run_path = tmp_path / "run"
        
        callbacks = exp.get_callbacks()
        
        assert isinstance(callbacks, list)
    
    def test_experiment_get_callbacks_with_checkpointing(self, base_vcfg: Config, tmp_path):
        """Includes ModelCheckpoint when log_checkpoints=True"""
        base_vcfg.logging.log_checkpoints = True
        base_vcfg.logging.save_top_k_models = 3
        base_vcfg.logging.create_folder = True
        
        exp = Experiment(base_vcfg)
        exp.run_path = tmp_path / "run"
        
        callbacks = exp.get_callbacks()
        
        # Should have at least one ModelCheckpoint callback
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_cbs = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(checkpoint_cbs) > 0


class TestExperimentTrainer:
    """Test get_trainer method."""
    
    def test_experiment_get_trainer(self, base_vcfg: Config, tmp_path):
        """Returns pl.Trainer instance"""
        exp = Experiment(base_vcfg)
        exp.run_path = tmp_path / "run"
        
        trainer = exp.get_trainer()
        
        assert trainer is not None
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "test")


class TestExperimentFit:
    """Test fit method."""
    
    def test_experiment_fit_workflow(self, base_vcfg: Config, tmp_path):
        """Full training workflow (mocked)"""
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        base_vcfg.training.trainer_params.max_epochs = 1
        base_vcfg.training.trainer_params.devices = 1
        base_vcfg.training.trainer_params.accelerator = "cpu"
        
        exp = Experiment(base_vcfg)
        
        # Mock trainer.fit to avoid actual training
        with patch("pytorch_lightning.Trainer.fit") as mock_fit:
            # Mock callback_metrics
            with patch("pytorch_lightning.Trainer.callback_metrics", new_callable=PropertyMock) as mock_metrics:
                mock_metrics.return_value = {base_vcfg.training.validation_metric: torch.tensor(0.5)}
                
                result = exp.fit()
                
                # Verify fit was called
                mock_fit.assert_called_once()
                
                # Verify experiment state
                assert exp.run_path is not None
                assert exp.model is not None
                assert exp.trainer is not None
                assert result == 0.5
    
    def test_experiment_fit_with_overrides(self, base_vcfg: Config, tmp_path):
        """Fit with config overrides"""
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        base_vcfg.training.trainer_params.max_epochs = 1
        base_vcfg.training.trainer_params.devices = 1
        base_vcfg.training.trainer_params.accelerator = "cpu"
        
        exp = Experiment(base_vcfg)
        
        overrides = {"training": {"trainer_params": {"max_epochs": 5}}}
        
        with patch("pytorch_lightning.Trainer.fit"):
            with patch("pytorch_lightning.Trainer.callback_metrics", new_callable=PropertyMock) as mock_metrics:
                mock_metrics.return_value = {base_vcfg.training.validation_metric: torch.tensor(0.3)}
                
                exp.fit(vcfg_overrides=overrides)
                
                assert exp.vcfg.training.trainer_params.max_epochs == 5


class TestExperimentTest:
    """Test test method."""
    
    def test_experiment_test_after_fit(self, base_vcfg: Config, tmp_path):
        """Test after fit with best model"""
        base_vcfg.logging.create_folder = True
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        base_vcfg.logging.log_checkpoints = True
        base_vcfg.logging.save_top_k_models = 1
        base_vcfg.training.trainer_params.max_epochs = 1
        base_vcfg.training.trainer_params.devices = 1
        base_vcfg.training.trainer_params.accelerator = "cpu"
        
        exp = Experiment(base_vcfg)
        
        # Mock fit and test
        with patch("pytorch_lightning.Trainer.fit"):
            with patch("pytorch_lightning.Trainer.test") as mock_test:
                with patch("pytorch_lightning.Trainer.callback_metrics", new_callable=PropertyMock) as mock_metrics:
                    mock_metrics.return_value = {base_vcfg.training.validation_metric: torch.tensor(0.5)}
                    
                    # First fit
                    exp.fit()
                    
                    # Mock checkpoint callback with PropertyMock
                    mock_checkpoint = MagicMock()
                    mock_checkpoint.best_model_path = str(tmp_path / "test_run" / "models" / "best.ckpt")
                    with patch.object(type(exp.trainer), "checkpoint_callback", new_callable=PropertyMock) as mock_cp_prop:
                        mock_cp_prop.return_value = mock_checkpoint
                        
                        # Then test
                        exp.test()
                        
                        # Verify test was called
                        mock_test.assert_called_once()
    
    def test_experiment_test_requires_checkpoint_name_without_fit(self, base_vcfg: Config, tmp_path):
        """Test without fit requires checkpoint_name"""
        base_vcfg.logging.create_folder = False
        base_vcfg.logging.run_path = str(tmp_path / "test_run")
        base_vcfg.training.trainer_params.devices = 1
        base_vcfg.training.trainer_params.accelerator = "cpu"
        
        exp = Experiment(base_vcfg)
        
        # Should raise error if no checkpoint_name provided
        with pytest.raises(ValueError, match="must provide a checkpoint name"):
            exp.test()


class TestExperimentCheckArgs:
    """Test _check_args validation."""
    
    def test_experiment_check_args_missing_dataset(self, base_vcfg: Config):
        """Raises error if dataset setup not defined"""
        base_vcfg.dataset.setup._target_ = "<dot.path.to.function>"
        
        with pytest.raises(ValueError, match="Dataset setup function is not defined"):
            Experiment(base_vcfg)
    
    def test_experiment_check_args_missing_model(self, base_vcfg: Config):
        """Raises error if model setup not defined"""
        base_vcfg.model.setup._target_ = "<dot.path.to.function>"
        
        with pytest.raises(ValueError, match="Model setup function is not defined"):
            Experiment(base_vcfg)
    
    def test_experiment_check_args_logging_without_folder(self, base_vcfg: Config):
        """Raises error if logging enabled but create_folder=False"""
        base_vcfg.logging.create_folder = False
        base_vcfg.logging.log_csv = True
        
        with pytest.raises(ValueError, match="enabled a logger but create_folder is False"):
            Experiment(base_vcfg)
