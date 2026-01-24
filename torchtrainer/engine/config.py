"""Configuration utilities for dynamic config management and object instantiation."""

import importlib
from collections.abc import Mapping
from copy import deepcopy
from functools import partial as functools_partial
from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.optim as optim
import yaml

# --- SHORTCUTS REGISTRY ---
# Reduces verbosity for standard PyTorch components.
SHORTCUTS = {
    # Optimizers
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    
    # Schedulers
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "StepLR": optim.lr_scheduler.StepLR,
    
    # Losses
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    
    # Common Layers
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "Sequential": nn.Sequential,
    "Conv2d": nn.Conv2d,
    "Flatten": nn.Flatten,
}

class Config(Mapping):
    """A dynamic configuration class that behaves like a dictionary but 
    allows dot-notation access. It supports loading from YAML, 
    updates, and saving back to YAML.
    """

    def __init__(self, path: str | Path | None = None):
        """Initializes the Config object.
        
        Args:
            path: Path to a YAML configuration file to initialize from.
                  If None, creates an empty config.
        """

        if path is None:
            return

        if isinstance(path, str | Path):
            self.update_from_yaml(path, allow_extra=True)
        elif isinstance(path, dict):
            self._load_from_dict(path)
        else:
            raise ValueError(
                "Config can only be initialized from a YAML file path or a dictionary.")

    @classmethod
    def _from_dict(cls, dictionary: dict[str, Any]) -> "Config":
        """Internal factory to create a Config instance from a dictionary.
        
        Args:
            dictionary: Dictionary to populate the new Config with.
            
        Returns:
            Config: A new Config instance populated with the dictionary data.
        """
        instance = cls()
        instance._load_from_dict(dictionary)
        return instance

    def _load_from_dict(self, dictionary: dict[str, Any]):
        """Internal helper to populate attributes from a dictionary.
        
        Recursively converts nested dictionaries to Config objects.
        
        Args:
            dictionary: Dictionary to populate attributes from.
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to Config objects using the factory
                value = Config._from_dict(value)
            
            setattr(self, key, value)

    def update_from_dict(self, dictionary: dict[str, Any], allow_extra: bool = False):
        """Recursively update the configuration using the given dictionary.

        Args:
            dictionary: Dictionary of values to merge into the current config.
            allow_extra: If False, raises ValueError if dictionary has keys not present in self.
        
        Raises:
            ValueError: If dictionary is not YAML-serializable or contains extra keys 
            when allow_extra=False.
        """
        # Check if dictionary is YAML serializable
        try:
            yaml.safe_dump(dictionary)
        except Exception:
            raise ValueError("Provided dictionary is not YAML-serializable") from None

        current_data = self.to_dict()
        
        if not allow_extra:
             self._validate_no_extra_keys(current_data, dictionary)

        updated_data = self._deep_update_dict(current_data, dictionary)
        
        # Clear current state and reload
        self.__dict__.clear()
        self._load_from_dict(updated_data)

    def update_from_yaml(self, path: str | Path, allow_extra: bool = False):
        """Recursively update the configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.
            allow_extra: If False, raises ValueError if YAML has keys not present in self.
            
        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If YAML contains extra keys when allow_extra=False.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {} # Handle empty YAML files safely

        current_data = self.to_dict()
        
        if not allow_extra:
             self._validate_no_extra_keys(current_data, data)

        updated_data = self._deep_update_dict(current_data, data)
        
        self.__dict__.clear()
        self._load_from_dict(updated_data)

    def save(self, path: str | Path):
        """Saves the current config state to a YAML file.
        
        Args:
            path: Path where the YAML file will be written.
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Recursively converts the Config object back to a standard dictionary.
        
        Required for YAML dumping.
        
        Returns:
            dict: Standard dictionary representation of the config.
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    # --- Utilities ---

    @staticmethod
    def _validate_no_extra_keys(base: dict, update: dict, prefix=""):
        """Helper to check if update contains keys not in base."""
        for key, value in update.items():
            if key not in base:
                raise ValueError(
                    f"Key '{prefix}{key}' is not present in "
                    "the current config and allow_extra=False.")
            
            if isinstance(value, dict) and isinstance(base[key], dict):
                Config._validate_no_extra_keys(base[key], value, prefix=f"{prefix}{key}.")

    @staticmethod
    def _deep_update_dict(base: dict, update: dict) -> dict:
        """Helper for recursive dictionary updates (in-place modification of base)."""
        for key, value in update.items():
            if (key in base and isinstance(base[key], dict) 
                    and isinstance(value, dict)):
                Config._deep_update_dict(base[key], value)
            else:
                base[key] = value
        return base
    
    def copy(self):
        """Creates a deep copy of the Config object."""
        # Now uses the factory method instead of init
        return Config._from_dict(deepcopy(self.to_dict()))

    def __repr__(self):
        return f"Config({self.to_dict()})"
    
    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def _repr_html_(self):
        return f"<pre>{self.__str__()}</pre>"

    def __getitem__(self, key):
        """Allow dictionary-style access: cfg["model"]."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in Config.")

    def __setitem__(self, key, value):
         """Allow dictionary-style setting: cfg["model"] = ..."""
         if isinstance(value, dict):
             # Recursively convert dict assignments to Config objects
             value = Config._from_dict(value)
         setattr(self, key, value)
         
    def __delitem__(self, key):
        """Allow dictionary-style deletion: del cfg["key"]."""
        if hasattr(self, key):
            delattr(self, key)
        else:
             raise KeyError(f"'{key}' not found in Config.")
             
    def __delattr__(self, name):
         """Allow removing attributes via del config.key."""
         try:
             super().__delattr__(name)
         except AttributeError:
             raise AttributeError(f"'{name}' not found in Config.") from None

    def __iter__(self):
        """Allows dict(config) and **config unpacking."""
        yield from self.__dict__

    def __len__(self):
        """Allows len(config)."""
        return len(self.__dict__)

    def __contains__(self, key):
        return hasattr(self, key)

def get_target(target_str: str):
    """Resolves a string to a Python class or function.

    If target_str is a registered shortcut, return it. Else, attempt to import it.
    """
    # 1. Shortcut Lookup
    if target_str in SHORTCUTS:
        return SHORTCUTS[target_str]
    
    # 2. Direct Import
    if "." in target_str:
        module_path, name = target_str.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{module_path}' for target '{target_str}'") from e
        try:
            obj = getattr(module, name)
        except AttributeError as e:
            raise AttributeError(f"Module '{module_path}' has no attribute '{name}'") from e
        return obj
    
    raise ValueError(
        f"Target '{target_str}' is not a registered shortcut and not a valid dot-path.")

def instantiate(config, partial: bool | None = None):
    """Recursively creates objects from a dictionary config.

    Args:
        config: The configuration object (Dict, List, or Config).
        partial: If True, forces return of a partial. 
              If False, forces instantiation. 
              If None (default), respects the '_partial_' key in the config.
    
    Special keywords:
    - _target_: The class or function to create.
    - _partial_: If True, returns a partial (factory) instead of an object.
    - _raw_: If True, returns the dictionary as-is (stops recursion).
    - keys starting with "_": Treated as meta keys and ignored during instantiation.
    """
    
    # 1. Handle Lists
    if isinstance(config, list):
        return [instantiate(item) for item in config]
    
    # 2. Handle Simple Values
    if not isinstance(config, Mapping):
        return config

    # 3. Handle Raw Configs (Stop recursion)
    # Works for both Config (via __getitem__) and dict
    if config.get("_raw_") is True:
        if hasattr(config, "to_dict"):
            clean = config.to_dict() 
        else:
            clean = config.copy()
        
        clean.pop("_raw_", None)
        return clean

    # 4. Standard Recursion (without target)
    if "_target_" not in config:
        return {k: instantiate(v) for k, v in config.items()}

    # --- INSTANTIATION LOGIC ---

    # A. Resolve Target
    target_str = config["_target_"]
    target = get_target(target_str)

    # B. Build Arguments (FILTERING HAPPENS HERE)
    kwargs = {}
    
    # We iterate over items filtering _* keys
    for k, v in config.items():
        if not k.startswith("_"):
            kwargs[k] = instantiate(v)
            
    # C. Check partial
    should_be_partial = partial if partial is not None else config.get("_partial_", False)
    
    if should_be_partial:
        return functools_partial(target, **kwargs)
    else:
        return target(**kwargs)