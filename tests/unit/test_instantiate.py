"""Unit tests for venturi.config instantiate and get_target functions."""

import pytest
import torch
import torch.nn as nn
from functools import partial

from venturi.config import Config, get_target, instantiate


# Test classes and functions for testing
class SimpleClass:
    """Simple test class."""
    def __init__(self, a, b=10):
        self.a = a
        self.b = b


class NestedClass:
    """Class that takes another class as argument."""
    def __init__(self, child):
        self.child = child


def simple_function(x, y=5):
    """Simple test function."""
    return x + y


class TestGetTarget:
    """Test get_target function."""

    def test_get_target_dotted_path_torch(self):
        """Resolve torch.nn.Linear style paths"""
        target = get_target("torch.nn.Linear")
        assert target is nn.Linear

    def test_get_target_dotted_path_torch_optim(self):
        """Resolve torch.optim.SGD style paths"""
        target = get_target("torch.optim.SGD")
        assert target is torch.optim.SGD

    def test_get_target_local_function(self):
        """Resolve function in local scope"""
        target = get_target("simple_function")
        assert target is simple_function

    def test_get_target_local_class(self):
        """Resolve class in local scope"""
        target = get_target("SimpleClass")
        assert target is SimpleClass

    def test_get_target_invalid_module(self):
        """Non-existent module raises ImportError"""
        with pytest.raises(ImportError, match="Could not import module"):
            get_target("nonexistent.module.Class")

    def test_get_target_invalid_attribute(self):
        """Module has no such attribute"""
        with pytest.raises(AttributeError, match="has no attribute"):
            get_target("torch.nn.NonExistentClass")

    def test_get_target_not_in_scope(self):
        """Object not found in stack frames"""
        with pytest.raises(NameError, match="not found"):
            get_target("CompletelyUnknownThing")


class TestInstantiateSimple:
    """Test instantiate with simple objects."""

    def test_instantiate_simple_class(self):
        """Instantiate class like torch.nn.Linear"""
        config = Config({
            "_target_": "torch.nn.Linear",
            "in_features": 10,
            "out_features": 5
        })
        
        obj = instantiate(config)
        assert isinstance(obj, nn.Linear)
        assert obj.in_features == 10
        assert obj.out_features == 5

    def test_instantiate_with_kwargs(self):
        """Pass keyword arguments"""
        config = Config({
            "_target_": "SimpleClass",
            "a": 100,
            "b": 200
        })
        
        obj = instantiate(config)
        assert isinstance(obj, SimpleClass)
        assert obj.a == 100
        assert obj.b == 200

    def test_instantiate_with_positional_args(self):
        """Use _args_ key"""
        config = Config({
            "_target_": "SimpleClass",
            "_args_": [100],
            "b": 200
        })
        
        obj = instantiate(config)
        assert isinstance(obj, SimpleClass)
        assert obj.a == 100
        assert obj.b == 200

    def test_instantiate_function(self):
        """Instantiate (call) a function"""
        config = Config({
            "_target_": "simple_function",
            "x": 10,
            "y": 20
        })
        
        result = instantiate(config)
        assert result == 30

    def test_instantiate_local_class(self):
        """Instantiate local test class"""
        config = Config({
            "_target_": "SimpleClass",
            "a": 42
        })
        
        obj = instantiate(config)
        assert isinstance(obj, SimpleClass)
        assert obj.a == 42
        assert obj.b == 10  # Default value


class TestInstantiatePartial:
    """Test instantiate with partial=True."""

    def test_instantiate_partial_true(self):
        """Create partial function"""
        config = Config({
            "_target_": "SimpleClass",
            "a": 100
        })
        
        factory = instantiate(config, partial=True)
        assert isinstance(factory, partial)
        
        # Call the factory
        obj = factory(b=300)
        assert isinstance(obj, SimpleClass)
        assert obj.a == 100
        assert obj.b == 300

    def test_instantiate_partial_false(self):
        """Force full instantiation"""
        config = Config({
            "_target_": "SimpleClass",
            "_partial_": True,  # Config says partial
            "a": 100
        })
        
        # But we override with partial=False
        obj = instantiate(config, partial=False)
        assert isinstance(obj, SimpleClass)
        assert not isinstance(obj, partial)

    def test_instantiate_partial_in_config(self):
        """Use _partial_ key in config"""
        config = Config({
            "_target_": "SimpleClass",
            "_partial_": True,
            "a": 100
        })
        
        factory = instantiate(config)
        assert isinstance(factory, partial)


class TestInstantiateNested:
    """Test instantiate with nested objects."""

    def test_instantiate_nested_objects(self):
        """Nested _target_ definitions"""
        config = Config({
            "_target_": "NestedClass",
            "child": {
                "_target_": "SimpleClass",
                "a": 42
            }
        })
        
        obj = instantiate(config)
        assert isinstance(obj, NestedClass)
        assert isinstance(obj.child, SimpleClass)
        assert obj.child.a == 42

    def test_instantiate_deeply_nested(self):
        """Multiple levels of nesting"""
        config = Config({
            "_target_": "NestedClass",
            "child": {
                "_target_": "NestedClass",
                "child": {
                    "_target_": "SimpleClass",
                    "a": 99
                }
            }
        })
        
        obj = instantiate(config)
        assert isinstance(obj, NestedClass)
        assert isinstance(obj.child, NestedClass)
        assert isinstance(obj.child.child, SimpleClass)
        assert obj.child.child.a == 99

    def test_instantiate_list_of_configs(self):
        """Instantiate list of objects"""
        config = [
            {
                "_target_": "SimpleClass",
                "a": 1
            },
            {
                "_target_": "SimpleClass",
                "a": 2
            }
        ]
        
        objects = instantiate(config)
        assert len(objects) == 2
        assert all(isinstance(obj, SimpleClass) for obj in objects)
        assert objects[0].a == 1
        assert objects[1].a == 2


class TestInstantiateSpecialKeys:
    """Test instantiate with special keys."""

    def test_instantiate_raw_flag(self):
        """Return raw dict when _raw_=True"""
        config = Config({
            "_target_": "SimpleClass",
            "a": 100,
            "_raw_": True
        })
        
        result = instantiate(config)
        assert isinstance(result, dict)
        assert not isinstance(result, Config)
        assert result == {"_target_": "SimpleClass", "a": 100}

    def test_instantiate_raw_removes_raw_key(self):
        """_raw_ key is removed from output"""
        config = Config({
            "data": "value",
            "_raw_": True
        })
        
        result = instantiate(config)
        assert "_raw_" not in result
        assert result == {"data": "value"}


class TestInstantiateNoTarget:
    """Test instantiate without _target_ key."""

    def test_instantiate_no_target_dict(self):
        """Dict without _target_ stays as Config or dict"""
        config = Config({
            "key1": "value1",
            "nested": {
                "key2": "value2"
            }
        })
        
        result = instantiate(config)
        assert isinstance(result, Config)
        assert result.key1 == "value1"
        assert isinstance(result.nested, Config)

    def test_instantiate_no_target_plain_dict(self):
        """Plain dict without _target_ stays as dict"""
        config = {
            "key1": "value1",
            "key2": "value2"
        }
        
        result = instantiate(config)
        assert isinstance(result, dict)
        assert result == config

    def test_instantiate_empty_dict(self):
        """Empty dict returns empty dict"""
        result = instantiate({})
        assert result == {}

    def test_instantiate_empty_config(self):
        """Empty Config returns empty Config"""
        config = Config()
        result = instantiate(config)
        assert isinstance(result, Config)
        assert len(result) == 0


class TestInstantiatePrimitives:
    """Test instantiate with primitive types."""

    def test_instantiate_string(self):
        """Return string unchanged"""
        result = instantiate("hello")
        assert result == "hello"

    def test_instantiate_int(self):
        """Return int unchanged"""
        result = instantiate(42)
        assert result == 42

    def test_instantiate_float(self):
        """Return float unchanged"""
        result = instantiate(3.14)
        assert result == 3.14

    def test_instantiate_bool(self):
        """Return bool unchanged"""
        result = instantiate(True)
        assert result is True

    def test_instantiate_none(self):
        """Return None unchanged"""
        result = instantiate(None)
        assert result is None

    def test_instantiate_list_of_primitives(self):
        """Return list of primitives unchanged"""
        data = [1, 2, "three", 4.0]
        result = instantiate(data)
        assert result == data


class TestInstantiateErrorHandling:
    """Test instantiate error handling."""

    def test_instantiate_invalid_target(self):
        """Non-existent class/function"""
        config = Config({
            "_target_": "NonExistentClass",
            "a": 1
        })
        
        with pytest.raises(NameError, match="not found"):
            instantiate(config)

    def test_instantiate_import_error(self):
        """Module doesn't exist"""
        config = Config({
            "_target_": "fake.module.Class",
            "a": 1
        })
        
        with pytest.raises(ImportError):
            instantiate(config)

    def test_instantiate_attribute_error(self):
        """Module has no such attribute"""
        config = Config({
            "_target_": "torch.nn.FakeLayer",
            "a": 1
        })
        
        with pytest.raises(AttributeError):
            instantiate(config)

    def test_instantiate_args_not_list(self):
        """_args_ is not a list"""
        config = Config({
            "_target_": "SimpleClass",
            "_args_": "not a list"
        })
        
        with pytest.raises(ValueError, match="_args_.*must be a list"):
            instantiate(config)

    def test_instantiate_missing_required_arg(self):
        """Missing required argument"""
        config = Config({
            "_target_": "SimpleClass"
            # Missing required 'a' argument
        })
        
        with pytest.raises(TypeError):
            instantiate(config)


class TestInstantiateRecursive:
    """Test recursive instantiation behavior."""

    def test_instantiate_recursive_configs(self):
        """Deep recursion through nested structures"""
        config = Config({
            "model": {
                "_target_": "torch.nn.Linear",
                "in_features": 10,
                "out_features": 5
            },
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "_partial_": True,
                "lr": 0.01
            },
            "metadata": {
                "name": "experiment",
                "nested": {
                    "value": 42
                }
            }
        })
        
        result = instantiate(config)
        
        # Model is instantiated
        assert isinstance(result.model, nn.Linear)
        
        # Optimizer is a partial
        assert isinstance(result.optimizer, partial)
        
        # Metadata without _target_ stays as Config
        assert isinstance(result.metadata, Config)
        assert result.metadata.name == "experiment"
        assert result.metadata.nested.value == 42

    def test_instantiate_mixed_list(self):
        """List with mix of targets and primitives"""
        config = [
            {"_target_": "SimpleClass", "a": 1},
            "plain string",
            42,
            {"no": "target"}
        ]
        
        result = instantiate(config)
        assert isinstance(result[0], SimpleClass)
        assert result[1] == "plain string"
        assert result[2] == 42
        assert isinstance(result[3], dict)


class TestInstantiateRealWorld:
    """Test instantiate with real PyTorch objects."""

    def test_instantiate_torch_linear(self):
        """Create torch.nn.Linear"""
        config = Config({
            "_target_": "torch.nn.Linear",
            "in_features": 128,
            "out_features": 10,
            "bias": True
        })
        
        layer = instantiate(config)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 128
        assert layer.out_features == 10

    def test_instantiate_torch_optimizer_partial(self):
        """Create SGD as partial"""
        config = Config({
            "_target_": "torch.optim.SGD",
            "_partial_": True,
            "lr": 0.01,
            "momentum": 0.9
        })
        
        optimizer_factory = instantiate(config)
        assert isinstance(optimizer_factory, partial)
        
        # Create optimizer with parameters
        model = nn.Linear(10, 5)
        optimizer = optimizer_factory(model.parameters())
        assert isinstance(optimizer, torch.optim.SGD)

    def test_instantiate_nested_torch_modules(self):
        """Create nested PyTorch modules"""
        # Sequential expects individual modules as arguments, not a list
        # We instantiate the modules first, then pass them to Sequential
        layer1 = instantiate({"_target_": "torch.nn.Linear", "in_features": 10, "out_features": 20})
        layer2 = instantiate({"_target_": "torch.nn.ReLU"})
        layer3 = instantiate({"_target_": "torch.nn.Linear", "in_features": 20, "out_features": 5})
        
        model = nn.Sequential(layer1, layer2, layer3)
        assert isinstance(model, nn.Sequential)
        assert len(model) == 3
        assert isinstance(model[0], nn.Linear)
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.Linear)
