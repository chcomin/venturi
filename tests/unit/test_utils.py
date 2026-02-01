"""Unit tests for venturi._util utility functions."""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from venturi.config import Config
from venturi._util import (
    generate_name_from_config,
    get_next_name,
    is_rank_zero,
    silence_lightning,
)


class TestGenerateNameFromConfig:
    """Test generate_name_from_config function."""
    
    def test_generate_name_simple(self):
        """Replace {key} placeholders"""
        config = Config({"lr": 0.001, "batch_size": 32})
        template = "lr_{lr}_bs_{batch_size}"
        
        result = generate_name_from_config(config, template)
        assert result == "lr_0.001_bs_32"
    
    def test_generate_name_nested(self):
        """Find nested config values"""
        # The function looks for keys recursively, not using dot notation
        # It searches for "name" and "lr" keys anywhere in the config
        config = Config({
            "model": {"name": "resnet"},
            "training": {"lr": 0.01}
        })
        template = "{name}_lr_{lr}"
        
        result = generate_name_from_config(config, template)
        assert result == "resnet_lr_0.01"
    
    def test_generate_name_formatting(self):
        """Apply format specs like {lr:.4f}"""
        config = Config({"lr": 0.00123456})
        template = "lr_{lr:.4f}"
        
        result = generate_name_from_config(config, template)
        assert result == "lr_0.0012"
    
    def test_generate_name_missing_key(self):
        """Raise KeyError for missing keys"""
        config = Config({"lr": 0.001})
        template = "lr_{lr}_bs_{batch_size}"
        
        with pytest.raises(KeyError):
            generate_name_from_config(config, template)
    
    def test_generate_name_no_placeholders(self):
        """Return template unchanged if no placeholders"""
        config = Config({"lr": 0.001})
        template = "my_experiment"
        
        result = generate_name_from_config(config, template)
        assert result == "my_experiment"


class TestGetNextName:
    """Test get_next_name function."""
    
    def test_get_next_name_file_not_exists(self, tmp_path):
        """Return original path"""
        path = tmp_path / "test.txt"
        result = get_next_name(path)
        assert result == path
    
    def test_get_next_name_file_exists(self, tmp_path):
        """Increment to _2, _3, etc."""
        path = tmp_path / "test.txt"
        path.write_text("content")
        
        result = get_next_name(path)
        assert result == tmp_path / "test_2.txt"
    
    def test_get_next_name_multiple_increments(self, tmp_path):
        """Handle multiple existing files"""
        (tmp_path / "test.txt").write_text("1")
        (tmp_path / "test_2.txt").write_text("2")
        (tmp_path / "test_3.txt").write_text("3")
        
        path = tmp_path / "test.txt"
        result = get_next_name(path)
        assert result == tmp_path / "test_4.txt"
    
    def test_get_next_name_directory(self, tmp_path):
        """Works with directories"""
        dir_path = tmp_path / "exp_folder"
        dir_path.mkdir()
        
        result = get_next_name(dir_path)
        assert result == tmp_path / "exp_folder_2"


class TestIsRankZero:
    """Test is_rank_zero function."""
    
    def test_is_rank_zero_true(self, monkeypatch):
        """Returns True when RANK=0"""
        monkeypatch.setenv("RANK", "0")
        assert is_rank_zero() is True
    
    def test_is_rank_zero_false(self, monkeypatch):
        """Returns False when RANK!=0"""
        monkeypatch.setenv("RANK", "1")
        assert is_rank_zero() is False
    
    def test_is_rank_zero_not_set(self, monkeypatch):
        """Returns True when RANK not set"""
        monkeypatch.delenv("RANK", raising=False)
        assert is_rank_zero() is True


class TestSilenceLightning:
    """Test silence_lightning function."""
    
    def test_silence_lightning(self):
        """Adds filters to loggers"""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            silence_lightning()
            
            # Should be called multiple times for different loggers
            assert mock_get_logger.called
            assert mock_logger.addFilter.called


class TestDeleteWandbRun:
    """Test delete_wandb_run function - only if wandb available."""
    
    @pytest.mark.skipif(not hasattr(pytest, "importorskip"), reason="Conditional test")
    def test_delete_wandb_run_mocked(self):
        """Delete run via API (mock)"""
        # This would require mocking the wandb API
        # For now, we'll just ensure the function exists and is callable
        from venturi._util import delete_wandb_run
        assert callable(delete_wandb_run)
