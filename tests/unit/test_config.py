"""Unit tests for venturi.config.Config class."""

import pickle
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from venturi.config import Config


class TestConfigInit:
    """Test Config initialization."""

    def test_config_init_empty(self):
        """Create empty Config()"""
        config = Config()
        assert len(config) == 0
        assert config.to_dict() == {}

    def test_config_init_from_dict(self):
        """Create Config from dictionary"""
        data = {"key1": "value1", "key2": 42}
        config = Config(data)
        assert config.key1 == "value1"
        assert config.key2 == 42
        assert len(config) == 2

    def test_config_init_from_yaml(self, tmp_path):
        """Create Config from YAML file"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\nkey2: 42\n")
        
        config = Config(yaml_file)
        assert config.key1 == "value1"
        assert config.key2 == 42

    def test_config_init_from_yaml_string_path(self, tmp_path):
        """Create Config from YAML file with string path"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key1: value1\n")
        
        config = Config(str(yaml_file))
        assert config.key1 == "value1"

    def test_config_init_invalid_source(self):
        """Pass invalid source type"""
        with pytest.raises(ValueError, match="can only be initialized"):
            Config(123)

    def test_config_nested_dict_auto_conversion(self):
        """Nested dicts become Config objects"""
        data = {"outer": {"inner": {"deep": "value"}}}
        config = Config(data)
        
        assert isinstance(config.outer, Config)
        assert isinstance(config.outer.inner, Config)
        assert config.outer.inner.deep == "value"


class TestConfigAccess:
    """Test Config access patterns."""

    def test_config_dot_notation_access(self):
        """Access nested values via config.key.subkey"""
        data = {"level1": {"level2": {"level3": "value"}}}
        config = Config(data)
        assert config.level1.level2.level3 == "value"

    def test_config_dict_notation_access(self):
        """Access values via config['key']['subkey']"""
        data = {"level1": {"level2": {"level3": "value"}}}
        config = Config(data)
        assert config["level1"]["level2"]["level3"] == "value"

    def test_config_mixed_access(self):
        """Mix dot and dict notation"""
        data = {"level1": {"level2": {"level3": "value"}}}
        config = Config(data)
        assert config.level1["level2"].level3 == "value"

    def test_config_getattr_missing_key(self):
        """Access non-existent key raises AttributeError"""
        config = Config({"existing": "value"})
        with pytest.raises(AttributeError, match="no attribute 'missing'"):
            _ = config.missing

    def test_config_key_conflicts_with_methods(self):
        """Keys named 'items', 'values', 'keys' accessible via dict syntax"""
        data = {"items": "test_value", "keys": "another_value"}
        config = Config(data)
        
        # Dict notation works
        assert config["items"] == "test_value"
        assert config["keys"] == "another_value"
        
        # Dot notation would return method, not data
        assert callable(config.items)  # This is the dict method
        assert callable(config.keys)

    def test_config_setitem(self):
        """Test __setitem__ behavior"""
        config = Config()
        config["key1"] = "value1"
        assert config.key1 == "value1"

    def test_config_setattr(self):
        """Test __setattr__ behavior"""
        config = Config()
        config.key1 = "value1"
        assert config["key1"] == "value1"

    def test_config_delitem(self):
        """Test __delitem__ behavior"""
        config = Config({"key1": "value1", "key2": "value2"})
        del config["key1"]
        assert "key1" not in config
        assert len(config) == 1

    def test_config_delattr(self):
        """Test __delattr__ behavior"""
        config = Config({"key1": "value1", "key2": "value2"})
        del config.key1
        assert "key1" not in config
        assert len(config) == 1


class TestConfigUpdate:
    """Test Config update methods."""

    def test_config_update_from_dict(self):
        """Update config with new dictionary"""
        config = Config({"key1": "value1", "nested": {"a": 1}})
        config.update_from_dict({"key2": "value2", "nested": {"b": 2}})
        
        assert config.key1 == "value1"
        assert config.key2 == "value2"
        assert config.nested.a == 1
        assert config.nested.b == 2

    def test_config_update_from_yaml(self, tmp_path):
        """Update config from YAML file"""
        config = Config({"key1": "value1"})
        
        yaml_file = tmp_path / "update.yaml"
        yaml_file.write_text("key2: value2\n")
        
        config.update_from_yaml(yaml_file)
        assert config.key1 == "value1"
        assert config.key2 == "value2"

    def test_config_update_from_config(self):
        """Merge two Config objects"""
        config1 = Config({"key1": "value1", "nested": {"a": 1}})
        config2 = Config({"key2": "value2", "nested": {"b": 2}})
        
        config1.update_from_config(config2)
        assert config1.key1 == "value1"
        assert config1.key2 == "value2"
        assert config1.nested.a == 1
        assert config1.nested.b == 2

    def test_config_update_from_source_dict(self):
        """Test update_from with dict source"""
        config = Config({"key1": "value1"})
        config.update_from({"key2": "value2"})
        assert config.key2 == "value2"

    def test_config_update_from_source_yaml(self, tmp_path):
        """Test update_from with YAML source"""
        config = Config({"key1": "value1"})
        yaml_file = tmp_path / "update.yaml"
        yaml_file.write_text("key2: value2\n")
        
        config.update_from(yaml_file)
        assert config.key2 == "value2"

    def test_config_update_from_source_config(self):
        """Test update_from with Config source"""
        config1 = Config({"key1": "value1"})
        config2 = Config({"key2": "value2"})
        config1.update_from(config2)
        assert config1.key2 == "value2"

    def test_config_update_from_invalid_source(self):
        """Test update_from with invalid source"""
        config = Config()
        with pytest.raises(ValueError, match="must be a YAML file path"):
            config.update_from(123)

    def test_config_update_allow_extra_false(self):
        """Reject keys not in original config"""
        config = Config({"key1": "value1"})
        
        with pytest.raises(ValueError, match="not present in the current config"):
            config.update_from_dict({"key2": "value2"}, allow_extra=False)

    def test_config_update_deeply_nested(self):
        """Update deeply nested structures"""
        config = Config({
            "level1": {
                "level2": {
                    "level3": {
                        "a": 1,
                        "b": 2
                    }
                }
            }
        })
        
        config.update_from_dict({
            "level1": {
                "level2": {
                    "level3": {
                        "b": 20,
                        "c": 3
                    }
                }
            }
        })
        
        assert config.level1.level2.level3.a == 1
        assert config.level1.level2.level3.b == 20
        assert config.level1.level2.level3.c == 3

    def test_config_update_non_yaml_serializable(self):
        """Pass unserializable dict"""
        config = Config()
        
        # Functions are not YAML serializable
        with pytest.raises(ValueError, match="not YAML-serializable"):
            config.update_from_dict({"key": lambda x: x})

    def test_config_load_missing_yaml(self):
        """FileNotFoundError for missing YAML"""
        config = Config()
        with pytest.raises(FileNotFoundError):
            config.update_from_yaml("nonexistent.yaml")

    def test_config_update_empty_yaml(self, tmp_path):
        """Handle empty YAML files"""
        config = Config({"key1": "value1"})
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        
        config.update_from_yaml(yaml_file)
        assert config.key1 == "value1"  # Original data preserved


class TestConfigSaveLoad:
    """Test Config save and load operations."""

    def test_config_save_to_yaml(self, tmp_path):
        """Save Config to YAML file"""
        config = Config({"key1": "value1", "nested": {"a": 1}})
        yaml_file = tmp_path / "output.yaml"
        
        config.save(yaml_file)
        
        assert yaml_file.exists()
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        assert data == {"key1": "value1", "nested": {"a": 1}}

    def test_config_yaml_roundtrip(self, tmp_path):
        """Save and reload config"""
        original = Config({"key1": "value1", "nested": {"a": 1, "b": [1, 2, 3]}})
        yaml_file = tmp_path / "roundtrip.yaml"
        
        original.save(yaml_file)
        loaded = Config(yaml_file)
        
        assert loaded.to_dict() == original.to_dict()


class TestConfigConversion:
    """Test Config conversion methods."""

    def test_config_to_dict(self):
        """Convert Config back to standard dict"""
        config = Config({"key1": "value1", "nested": {"a": 1}})
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert not isinstance(result, Config)
        assert result == {"key1": "value1", "nested": {"a": 1}}

    def test_config_to_dict_deeply_nested(self):
        """Ensure all nested Configs become dicts"""
        config = Config({"l1": {"l2": {"l3": "value"}}})
        result = config.to_dict()
        
        assert isinstance(result["l1"], dict)
        assert not isinstance(result["l1"], Config)
        assert isinstance(result["l1"]["l2"], dict)


class TestConfigCopy:
    """Test Config copy operations."""

    def test_config_copy(self):
        """Deep copy a Config object"""
        original = Config({"key1": "value1", "nested": {"a": 1}})
        copied = original.copy()
        
        # Modify copy
        copied.nested.a = 999
        copied.key2 = "value2"
        
        # Original unchanged
        assert original.nested.a == 1
        assert "key2" not in original

    def test_config_deepcopy(self):
        """Ensure deepcopy works correctly"""
        original = Config({"key1": "value1", "nested": {"a": 1}})
        copied = deepcopy(original)
        
        # Modify copy
        copied.nested.a = 999
        
        # Original unchanged
        assert original.nested.a == 1

    def test_config_pickle_unpickle(self):
        """Test serialization/deserialization"""
        original = Config({"key1": "value1", "nested": {"a": 1}})
        
        # Pickle
        pickled = pickle.dumps(original)
        
        # Unpickle
        restored = pickle.loads(pickled)
        
        assert restored.to_dict() == original.to_dict()
        assert restored.nested.a == 1


class TestConfigMappingInterface:
    """Test MutableMapping interface."""

    def test_config_iter(self):
        """Test iteration over keys"""
        config = Config({"a": 1, "b": 2, "c": 3})
        keys = list(config)
        assert set(keys) == {"a", "b", "c"}

    def test_config_len(self):
        """Test len() function"""
        config = Config({"a": 1, "b": 2})
        assert len(config) == 2

    def test_config_contains(self):
        """Test 'in' operator"""
        config = Config({"a": 1, "b": 2})
        assert "a" in config
        assert "c" not in config

    def test_config_items(self):
        """Test items() method"""
        config = Config({"a": 1, "b": 2})
        items = dict(config.items())
        assert items == {"a": 1, "b": 2}

    def test_config_keys(self):
        """Test keys() method"""
        config = Config({"a": 1, "b": 2})
        assert set(config.keys()) == {"a", "b"}

    def test_config_values(self):
        """Test values() method"""
        config = Config({"a": 1, "b": 2})
        assert set(config.values()) == {1, 2}


class TestConfigStringRepresentations:
    """Test string representations."""

    def test_config_repr(self):
        """Test __repr__"""
        config = Config({"key": "value"})
        repr_str = repr(config)
        assert "Config" in repr_str
        assert "key" in repr_str

    def test_config_str(self):
        """Test __str__"""
        config = Config({"key": "value"})
        str_repr = str(config)
        assert "key: value" in str_repr

    def test_config_repr_html(self):
        """Test _repr_html_ for Jupyter"""
        config = Config({"key": "value"})
        html = config._repr_html_()
        assert "<pre>" in html
        assert "key: value" in html


class TestConfigEdgeCases:
    """Test edge cases and error conditions."""

    def test_config_empty_update(self):
        """Update with empty dict does nothing"""
        config = Config({"key": "value"})
        config.update_from_dict({})
        assert config.key == "value"

    def test_config_overwrite_nested_with_scalar(self):
        """Overwrite nested config with scalar value"""
        config = Config({"nested": {"a": 1, "b": 2}})
        config.update_from_dict({"nested": "scalar"})
        assert config.nested == "scalar"

    def test_config_overwrite_scalar_with_nested(self):
        """Overwrite scalar with nested config"""
        config = Config({"value": "scalar"})
        config.update_from_dict({"value": {"a": 1}})
        assert isinstance(config.value, Config)
        assert config.value.a == 1

    def test_config_nested_extra_keys_validation(self):
        """Test allow_extra=False with nested structures"""
        config = Config({"nested": {"a": 1}})
        
        with pytest.raises(ValueError, match="nested.b.*not present"):
            config.update_from_dict({"nested": {"b": 2}}, allow_extra=False)

    def test_config_setitem_converts_dict_to_config(self):
        """Setting a dict value auto-converts to Config"""
        config = Config()
        config["nested"] = {"a": 1}
        assert isinstance(config.nested, Config)
        assert config.nested.a == 1

    def test_config_setattr_converts_dict_to_config(self):
        """Setting a dict via dot notation auto-converts to Config"""
        config = Config()
        config.nested = {"a": 1}
        assert isinstance(config.nested, Config)
        assert config.nested.a == 1
