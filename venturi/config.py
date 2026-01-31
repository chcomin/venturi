"""Configuration utilities for dynamic config management and object instantiation."""

import argparse
import importlib
import inspect
import shutil
import sys
from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from functools import partial as functools_partial
from importlib import resources
from pathlib import Path
from typing import Any, Self

import yaml


class Config(MutableMapping):
    """A dynamic configuration class that behaves like a dictionary but allows dot-notation
    access. It supports loading from YAML, updating from another Config, and saving back to
    YAML.

    Data is stored in a private dictionary (_data) to prevent conflicts between config keys
    (like 'items', 'values') and class methods. Note: If a key conflicts with a method name, it
    is only accessible via dictionary syntax (config['items']) and not dot notation
    (config.items).
    """

    def __init__(self, source: str | Path | dict | None = None):
        """Initializes the Config object.

        Args:
            source: Path to a YAML configuration file or a dictionary to initialize from.
            If None, creates an empty config.
        """
        super().__setattr__("_data", {})

        if source is None:
            return

        if isinstance(source, str | Path):
            self.update_from_yaml(source, allow_extra=True)
        elif isinstance(source, dict):
            self._load_from_dict(source)
        else:
            raise ValueError(
                "Config can only be initialized from a YAML file path or a dictionary."
            )

    @classmethod
    def _from_dict(cls, dictionary: dict[str, Any]) -> Self:
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
        data_store = self.__dict__["_data"]
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts to Config objects using the factory
                value = Config._from_dict(value)

            data_store[key] = value

    def update_from_dict(self, dictionary: dict[str, Any], allow_extra: bool = True):
        """Recursively update the configuration using the given dictionary. Existing keys are
        updated with new values, while new keys are added if allow_extra=True.

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
        self.__dict__["_data"].clear()
        self._load_from_dict(updated_data)

    def update_from_yaml(self, path: str | Path, allow_extra: bool = True):
        """Recursively update the configuration from a YAML file. Existing keys are
        updated with new values, while new keys are added if allow_extra=True.

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
            data = yaml.safe_load(f) or {}  # Handle empty YAML files safely

        current_data = self.to_dict()

        if not allow_extra:
            self._validate_no_extra_keys(current_data, data)

        updated_data = self._deep_update_dict(current_data, data)

        self.__dict__["_data"].clear()
        self._load_from_dict(updated_data)

    def update_from_config(self, other: Self, allow_extra: bool = True):
        """Recursively update the configuration from another Config object.

        Args:
            other: Config object to merge into the current config.
            allow_extra: If False, raises ValueError if other has keys not present in self.
        """
        self.update_from_dict(other.to_dict(), allow_extra=allow_extra)

    def update_from(self, source: str | Path | dict[str, Any] | Self, allow_extra: bool = True):
        """Recursively update the configuration from a source. Existing keys are updated with
        new values, while new keys are added if allow_extra=True.

        Args:
            source: Source to update from. Can be a YAML file path, dictionary, or Config object.
            allow_extra: If False, raises ValueError if source has keys not present in self.
        """
        if isinstance(source, str | Path):
            self.update_from_yaml(source, allow_extra=allow_extra)
        elif isinstance(source, dict):
            self.update_from_dict(source, allow_extra=allow_extra)
        elif isinstance(source, Config):
            self.update_from_config(source, allow_extra=allow_extra)
        else:
            raise ValueError("Source must be a YAML file path, dictionary, or Config object.")

    def save(self, path: str | Path):
        """Saves the current config state to a YAML file.

        Args:
            path: Path where the YAML file will be written.
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """Recursively converts the Config object back to a standard dictionary.

        Returns:
            dict: Standard dictionary representation of the config.
        """
        result = {}
        for key, value in self.__dict__.get("_data", {}).items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    # --- Mapping Interface Implementation ---

    def __getitem__(self, key: str) -> Any:
        return self.__dict__["_data"][key]

    def __setitem__(self, key: str, value: Any):
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config._from_dict(value)
        self.__dict__["_data"][key] = value

    def __delitem__(self, key: str):
        del self.__dict__["_data"][key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__.get("_data", {}))

    def __len__(self) -> int:
        return len(self.__dict__.get("_data", {}))

    def __contains__(self, key: Any) -> bool:
        return key in self.__dict__.get("_data", {})

    # --- Dot Notation Access ---

    def __getattr__(self, name: str) -> Any:
        """Called only if attribute lookup failed in __dict__.

        Note: Must NOT access self._data here via dot notation,
        as that triggers __getattr__ recursively if _data is missing.
        """
        data = self.__dict__.get("_data")

        # If _data is missing (e.g. during unpickling/init), we cannot find the key.
        if data is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' (Config not initialized)"
            )

        # Look up the key in the data dictionary
        try:
            return data[name]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from e

    def __setattr__(self, name: str, value: Any):
        """Handle attribute setting logic."""
        if name == "_data":
            super().__setattr__(name, value)
            return

        # If it's a known non-config attribute (exists in __dict__), update it there.
        if name in self.__dict__:
            super().__setattr__(name, value)
            return

        data = self.__dict__.get("_data")

        # If _data is initialized, store it there
        if data is not None:
            # We use __setitem__ logic manually to ensure recursion safety
            if isinstance(value, dict) and not isinstance(value, Config):
                value = Config._from_dict(value)
            data[name] = value
        else:
            # Fallback for uninitialized objects (prevents crash during weird state loading)
            super().__setattr__(name, value)

    def __delattr__(self, name: str):
        data = self.__dict__.get("_data")
        if data and name in data:
            del data[name]
        else:
            super().__delattr__(name)

    # --- Pickling Support (Fixes Deepcopy issues) ---

    def __getstate__(self):
        """Custom pickle state."""
        return self.__dict__

    def __setstate__(self, state):
        """Custom unpickle state."""
        self.__dict__.update(state)
        # Ensure _data exists after unpickling
        if "_data" not in self.__dict__:
            super().__setattr__("_data", {})

    # --- Utilities ---

    @staticmethod
    def _validate_no_extra_keys(base: dict, update: dict, prefix=""):
        """Helper to check if update contains keys not in base."""
        for key, value in update.items():
            if key not in base:
                raise ValueError(
                    f"Key '{prefix}{key}' is not present in "
                    "the current config and allow_extra=False."
                )

            if isinstance(value, dict) and isinstance(base[key], dict):
                Config._validate_no_extra_keys(base[key], value, prefix=f"{prefix}{key}.")

    @staticmethod
    def _deep_update_dict(base: dict, update: dict) -> dict:
        """Helper for recursive dictionary updates (in-place modification of base)."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_update_dict(base[key], value)
            else:
                base[key] = value
        return base

    def copy(self):
        """Creates a deep copy of the Config object."""
        return Config._from_dict(deepcopy(self.to_dict()))

    def __repr__(self):
        return f"Config({self.__dict__.get('_data', {})})"

    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def _repr_html_(self):
        return f"<pre>{self.__str__()}</pre>"


def get_target(target_str: str) -> Any:
    """Resolves a string to a Python class or function."""

    if "." in target_str:
        # If a dotted path, import the module and get the object using getattr
        module_path, name = target_str.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(f"Could not import module '{module_path}'") from e

        try:
            return getattr(module, name)
        except AttributeError as e:
            raise AttributeError(f"Module '{module_path}' has no attribute '{name}'") from e

    # target_str is not a dotted path, we need to search for it on stack frames
    frame = inspect.currentframe()
    try:
        while frame:
            # Check locals
            if target_str in frame.f_locals:
                return frame.f_locals[target_str]

            # Check globals
            if target_str in frame.f_globals:
                return frame.f_globals[target_str]

            frame = frame.f_back
    finally:
        del frame

    raise NameError(f"Object '{target_str}' not found. Is the name correct and in scope?")


def instantiate(config: Config | dict | list | Any, partial: bool | None = None) -> Any:
    """Recursively creates objects from a given Config object. Objects to be instantiated must
    have a '_target_' key specifying the class or function to create. The remaining keys are treated
    as arguments to the target's constructor or factory function.

    Args:
        config: The configuration object.
        partial: If True, forces return of a partial. That is, a factory function that can be
        called later to create the object. If False, forces instantiation. If None (default),
        respects the '_partial_' key in the config.

    Special keywords:
    - _target_: The class or function to create. Can be a dotted path or a name in scope.
    - _partial_: If True, returns a partial (factory).
    - _args_: List of positional arguments to pass to the target.
    - _raw_: If True, returns the config as-is (stops recursion).
    """

    # Handle lists
    if isinstance(config, list):
        return [instantiate(item) for item in config]  

    # Handle simple values (primitives)
    if not isinstance(config, (Config, dict)):
        return config

    # Helper to get items whether it's Config or dict
    # If it's a Config, to_dict() would recurse, so we access items directly.
    # Config implements MutableMapping, so .items() works and yields Config/dict children.

    # Check for _raw_ flag
    # We access safely to avoid triggering __getattr__ logic excessively
    is_raw = config.get("_raw_")

    if is_raw is True:
        if isinstance(config, Config):
            clean = config.to_dict()
        else:
            clean = deepcopy(config)

        clean.pop("_raw_", None)
        return clean

    # If no target, recurse but preserve the container type (Config)
    # Check existence safely
    has_target = "_target_" in config

    if not has_target:
        # Recursively instantiate children
        instantiated_data = {k: instantiate(v) for k, v in config.items()}

        # Return Config object if input was Config
        if isinstance(config, Config):
            return Config._from_dict(instantiated_data)
        return instantiated_data

    # --- INSTANTIATION LOGIC ---

    # Resolve Target
    target_str = config["_target_"]
    target = get_target(target_str)

    # Build arguments
    kwargs = {}
    args = []

    for k, v in config.items():
        if k == "_args_":
            # Positional arguments
            if not isinstance(v, list):
                raise ValueError(f"'_args_' must be a list, got {type(v)}")
            args = [instantiate(arg) for arg in v]
            continue

        if k not in ("_target_", "_partial_", "_args_", "_raw_"):
            kwargs[k] = instantiate(v)

    # Check if partial
    # prioritize function arg 'partial', then config '_partial_', default False
    should_be_partial = partial if partial is not None else config.get("_partial_", False)

    if should_be_partial:
        return functools_partial(target, *args, **kwargs)
    else:
        return target(*args, **kwargs)


def create_config(args_list: list[str] | None = None):
    """CLI to create a new Venturi configuration file.

    Args:
        args_list: Optional list of arguments (for testing).
                   Defaults to sys.argv[1:].
    """

    parser = argparse.ArgumentParser(prog="venturi", description="Venturi CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    create_parser = subparsers.add_parser("create", help="Create a new configuration file.")

    create_parser.add_argument(
        "destination_path",
        type=str,
        nargs="?",
        default=".",
        help="Folder where the configuration file will be saved. Defaults to current directory.",
    )

    if args_list is None:
        args_list = sys.argv[1:]

    if not args_list:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args_list)
    if args.command == "create":
        try:
            with resources.path("venturi", "base_config.yaml") as p:
                cfg_path = p
        except (ImportError, FileNotFoundError):
            # Fallback for local development (not installed as package)
            current_dir = Path(__file__).resolve().parent
            cfg_path = current_dir / "base_config.yaml"

        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Source config file missing at {cfg_path}. Check installation."
            )

        dest_dir = Path(args.destination_path).resolve()

        # Check if user accidentally provided a filename ending in .yaml
        if dest_dir.suffix in [".yaml", ".yml"] and not dest_dir.exists():
            print(
                f"Warning: '{dest_dir.name}' looks like a file, but this command expects a folder."
            )
            print(
                f"I will create a FOLDER named '{dest_dir.name}' and put base_config.yaml "
                "inside it."
            )
            dest_dir = dest_dir.parent / dest_dir.stem

        if not dest_dir.exists():
            dest_dir.mkdir(parents=True)

        target_file = dest_dir / "base_config.yaml"

        if target_file.exists():
            print(f"Error: Configuration file already exists at {target_file}")
            overwrite = input("Do you want to overwrite it? (This cannot be undone) [y/N]: ")
            if overwrite.lower() != "y":
                print("Operation cancelled.")
                sys.exit(0)

        try:
            shutil.copy(cfg_path, target_file)
            print(f"Configuration file created at: {target_file}")
        except PermissionError:
            print(f"Error: Permission denied. Cannot write to {target_file}")
            sys.exit(1)
