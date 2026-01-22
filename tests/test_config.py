from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import torch.nn as nn
import yaml

from torchtrainer.engine.config import Config, get_target, instantiate


class DummyModel:
    """A simple class to test instantiation."""
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def __eq__(self, other):
        return (self.in_features == other.in_features and 
                self.out_features == other.out_features and
                self.bias == other.bias)

def dummy_factory(a, b):
    """A simple function to test functional instantiation."""
    return a + b

# --- Fixtures ---

@pytest.fixture
def basic_dict():
    return {
        "model": {
            "type": "cnn",
            "layers": [16, 32, 64]
        },
        "training": {
            "lr": 0.001,
            "epochs": 10
        }
    }

@pytest.fixture
def yaml_file(basic_dict):
    """Creates a temporary YAML file."""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(basic_dict, f)
        path = f.name
    yield path
    # Cleanup
    Path(path).unlink(missing_ok=True)

# --- Tests for Config Class ---

class TestConfig:
    
    def test_init_empty(self):
        cfg = Config()
        assert len(cfg) == 0
        assert isinstance(cfg, Config)

    def test_from_dict_and_access(self, basic_dict):
        # Test loading via internal factory used by update methods
        cfg = Config()
        # FIX: Must allow extra keys when populating empty config
        cfg.update_from_dict(basic_dict, allow_extra=True)
        
        # Dot access
        assert cfg.training.lr == 0.001
        assert cfg.model.type == "cnn"
        
        # Dict access
        assert cfg["training"]["epochs"] == 10
        
        # Nested Type check
        assert isinstance(cfg.model, Config)

    def test_update_from_dict_merge(self):
        cfg = Config()
        # FIX: Must allow extra keys for initial population
        cfg.update_from_dict({"a": 1, "nested": {"x": 10}}, allow_extra=True)
        
        update_data = {"b": 2, "nested": {"y": 20}}
        cfg.update_from_dict(update_data, allow_extra=True)
        
        assert cfg.a == 1
        assert cfg.b == 2
        assert cfg.nested.x == 10  # Kept existing
        assert cfg.nested.y == 20  # Added new

    def test_update_no_extra_keys_error(self):
        cfg = Config()
        # FIX: Must allow extra keys for initial population
        cfg.update_from_dict({"a": 1, "nested": {"x": 10}}, allow_extra=True)
        
        # Should fail because 'b' is new and default is allow_extra=False
        with pytest.raises(ValueError, match="Key 'b' is not present"):
            cfg.update_from_dict({"b": 2}, allow_extra=False)

        # Should fail because 'nested.y' is new
        with pytest.raises(ValueError, match="Key 'nested.y' is not present"):
            cfg.update_from_dict({"nested": {"y": 20}}, allow_extra=False)

    def test_update_yaml(self, yaml_file):
        cfg = Config(yaml_file)
        assert cfg.training.lr == 0.001
        
        # Test update from yaml
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"training": {"lr": 0.05}}, f)
            update_path = f.name
            
        try:
            cfg.update_from_yaml(update_path)
            assert cfg.training.lr == 0.05
            assert cfg.training.epochs == 10 # Should preserve old value
        finally:
            Path(update_path).unlink(missing_ok=True)

    def test_save_and_load(self, basic_dict):
        cfg = Config()
        cfg.update_from_dict(basic_dict, allow_extra=True)
        
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            save_path = f.name
            
        try:
            cfg.save(save_path)
            # Reload to verify
            new_cfg = Config(save_path)
            assert new_cfg.to_dict() == basic_dict
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_copy(self):
        cfg = Config()
        cfg.update_from_dict({"a": [1, 2], "b": {"c": 3}}, allow_extra=True)
        
        cfg_copy = cfg.copy()
        
        # Modify copy
        cfg_copy.a.append(3)
        cfg_copy.b.c = 99
        
        # Original should be untouched
        assert cfg.a == [1, 2]
        assert cfg.b.c == 3
        assert cfg_copy.a == [1, 2, 3]

    def test_magic_methods(self):
        cfg = Config()
        cfg.val = 100
        
        # __contains__
        assert "val" in cfg
        assert "nonexistent" not in cfg
        
        # __setitem__
        cfg["new_val"] = 200
        assert cfg.new_val == 200
        
        # __delitem__ / __delattr__
        del cfg["val"]
        assert "val" not in cfg
        with pytest.raises(KeyError):
            del cfg["val"]

    def test_repr_and_str(self):
        cfg = Config()
        cfg.a = 1
        assert "Config" in repr(cfg)
        assert "a: 1" in str(cfg)

# --- Tests for instantiation logic ---

class TestInstantiation:

    def test_get_target_shortcut(self):
        cls = get_target("Linear")
        assert cls == nn.Linear

    def test_get_target_import(self):
        # Test importing a standard library function
        func = get_target("math.cos")
        import math
        assert func == math.cos

    def test_get_target_errors(self):
        with pytest.raises(ValueError):
            get_target("InvalidNameWithoutDot")
            
        with pytest.raises(ImportError):
            get_target("non_existent_module.Class")
            
        with pytest.raises(AttributeError):
            get_target("math.non_existent_function")

    def test_instantiate_class(self):
        config = {
            "_target_": "test_config.DummyModel",
            "in_features": 10,
            "out_features": 5,
            "bias": False
        }
        obj = instantiate(config)
        assert isinstance(obj, DummyModel)
        assert obj.in_features == 10
        assert obj.bias is False

    def test_instantiate_recursive(self):
        # A list of objects
        config = [
            {"_target_": "test_config.dummy_factory", "a": 1, "b": 2},
            {"_target_": "test_config.dummy_factory", "a": 10, "b": 20},
        ]
        objs = instantiate(config)
        assert objs == [3, 30]

    def test_instantiate_lazy(self):
        config = {
            "_target_": "test_config.dummy_factory",
            "_partial_": True,
            "a": 5
        }
        partial_obj = instantiate(config)
        assert isinstance(partial_obj, partial)
        # Call the partial to verify it works
        assert partial_obj(b=5) == 10

    def test_instantiate_raw(self):
        config = {
            "_target_": "should.not.be.Imported",
            "_raw_": True,
            "some_data": 123
        }
        result = instantiate(config)
        assert isinstance(result, dict)
        assert result["_target_"] == "should.not.be.Imported"
        assert "_raw_" not in result
        assert result["some_data"] == 123

    def test_instantiate_meta_keys_ignored(self):
        """Test that keys starting with _ (but not reserved ones) are ignored."""
        config = {
            "_target_": "test_config.dummy_factory",
            "a": 1,
            "b": 1,
            "_comment": "This should be ignored",
            "_version": 1.0
        }
        # dummy_factory only accepts a and b. If _comment is passed, it raises TypeError
        res = instantiate(config)
        assert res == 2

    def test_instantiate_config_object(self):
        """Ensure instantiate works with Config objects, not just dicts."""
        cfg = Config()
        cfg._target_ = "test_config.DummyModel"
        cfg.in_features = 2
        cfg.out_features = 2
        
        obj = instantiate(cfg)
        assert isinstance(obj, DummyModel)

    def test_instantiate_shortcuts(self):
        """Test the PyTorch shortcuts."""
        config = {
            "_target_": "Linear",
            "in_features": 10,
            "out_features": 5
        }
        obj = instantiate(config)
        assert isinstance(obj, nn.Linear)
        assert obj.in_features == 10
        assert obj.out_features == 5

    def test_instantiate_lazy_override_true(self):
        """Test that lazy=True argument overrides _lazy_: False in config."""
        config = {
            "_target_": "test_config.dummy_factory",
            "_partial_": False, # Config says instantiate immediately
            "a": 5
        }
        
        # Code requests lazy=True, should get a partial
        partial_obj = instantiate(config, partial=True)
        assert isinstance(partial_obj, partial)
        assert partial_obj(b=5) == 10

    def test_instantiate_lazy_override_false(self):
        """Test that lazy=False argument overrides _lazy_: True in config."""
        config = {
            "_target_": "test_config.dummy_factory",
            "_partial_": True, # Config says be lazy
            "a": 5,
            "b": 5
        }
        
        # Code requests lazy=False, should force instantiation
        result = instantiate(config, partial=False)
        assert not isinstance(result, partial)
        assert result == 10

    def test_instantiate_lazy_override_none(self):
        """Test that lazy=None falls back to config."""
        config_lazy = {
            "_target_": "test_config.dummy_factory",
            "_partial_": True,
            "a": 5
        }
        assert isinstance(instantiate(config_lazy, partial=None), partial)

        config_strict = {
            "_target_": "test_config.dummy_factory",
            "_partial_": False,
            "a": 5, 
            "b": 5
        }
        assert not isinstance(instantiate(config_strict, partial=None), partial)

if __name__ == "__main__":
    pytest.main([__file__])