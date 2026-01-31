import pytest
import torch.nn as nn

from venturi.config import Config


@pytest.fixture
def basic_config(tmp_path):
    """Returns a minimal valid Config object."""
    # tmp_path is a built-in pytest fixture that creates a temp folder
    # that is auto-deleted after the test.
    return Config()


@pytest.fixture
def simple_cls_model():
    return nn.Linear(10, 2)


@pytest.fixture
def simple_seg_model():
    return nn.Conv2d(1, 1, 3, padding=1)
