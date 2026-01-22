""" Pydantic configuration classes. """

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Schema(BaseModel):
    """Base class for validating configuration parameters."""

    model_config = ConfigDict(
        extra="forbid",            
        validate_assignment=True,       # Useful when updating fields
        arbitrary_types_allowed=False,  # Guarantees serialization works
        str_strip_whitespace=True,
        protected_namespaces=(),    # Important! By default Pydantic does not allow model_* fields
        use_enum_values=True,
        validate_default=True
    )

    def __init__(self, **data):
        super().__init__(**data)
        self._check_pydantic_collisions()

    def _check_pydantic_collisions(self):
        """Check if any user-defined fields shadow attributes/methods 
        that exist on the standard Pydantic BaseModel.
        """
        # Get all attributes of a raw BaseModel (methods, properties, etc.)
        if not hasattr(Schema, "_pydantic_reserved_names"):
            Schema._pydantic_reserved_names = set(dir(BaseModel))

        # Get the fields defined in the current instance
        current_fields = set(self.__class__.model_fields.keys())

        # Find collisions
        collisions = current_fields.intersection(Schema._pydantic_reserved_names)
        
        if collisions:
            raise NameError(
                f"Configuration Safety Error: The following fields defined in "
                f"'{self.__class__.__name__}' conflict with Pydantic internal methods: "
                f"{collisions}. Please rename them."
            )

    def deep_update(self, update_dict: dict[str, Any]) -> Schema:
        """Recursively updates the config with a dictionary."""
        
        for key, value in update_dict.items():
            if not hasattr(self, key):
                # Because extra="forbid", this check is redundant for assignment 
                # but useful for skipping invalid keys before they raise an error.
                continue

            current_value = getattr(self, key)

            # Recurse if both are Pydantic models (or BaseConfig) and value is dict
            if isinstance(current_value, BaseModel) and isinstance(value, dict):
                # If the nested object has deep_update, use it (BaseConfig check)
                if hasattr(current_value, "deep_update"):
                    current_value.deep_update(value)
                else:
                    # Fallback for standard Pydantic models
                    self._recursive_update_standard(current_value, value)
            else:
                # Direct assignment (triggers Pydantic validation)
                setattr(self, key, value)
        
        return self

    def _recursive_update_standard(self, model: BaseModel, update_data: dict):
        """Helper for standard Pydantic models not inheriting from BaseConfig."""

        for k, v in update_data.items():
            if hasattr(model, k):
                attr = getattr(model, k)
                if isinstance(attr, BaseModel) and isinstance(v, dict):
                    self._recursive_update_standard(attr, v)
                else:
                    setattr(model, k, v)

class LoggingParams(Schema):
    """Logging and experiment related parameters."""

    experiments_path: Path = Field(
        default=Path("experiments"), 
        description="Path to save experiments data"
    )
    experiment_name: str = Field(
        default="no_name_experiment", 
        description="Name of the experiment"
    )
    run_name: str = Field(
        default="no_name_run", 
        description="Name of the run for a given experiment"
    )
    validate_every: int = Field(
        default=1, ge=1, 
        description="Run a validation step every N epochs"
    )
    save_val_imgs: bool = Field(
        default=False, 
        description="Save some validation images when validating"
    )
    val_img_indices: list[int] = Field(
        default=[0], 
        description="Indices of the validation images to save"
    )
    suppress_checkpoint: bool = Field(
        default=False, 
        description="Suppress model checkpoint saving at the end of each epoch."
    )
    copy_model_every: int = Field(
        default=0, ge=0, 
        description="Save a copy of the model every N epochs. If 0, no copies are saved."
    )
    suppress_best_checkpoint: bool = Field(
        default=False, 
        description="Avoid saving the best checkpoint found during training."
    )
    log_wandb: bool = Field(
        default=False, 
        description="If wandb should also be used for logging."
    )
    wandb_project: str = Field(
        default="uncategorized", 
        description="Name of the wandb project to log the data."
    )
    wandb_group: list[str] = Field(
        default_factory=list, 
        description="Name of the wandb group to log the data."
    )
    log_images_wandb: bool = Field(
        default=False, 
        description="If wandb should be used for logging validation images."
    )
    disable_tqdm: bool = Field(
        default=False, 
        description="Disable tqdm progressbar."
    )
    meta: list[str] = Field(
        default_factory=list, 
        description="Additional metadata to save in the config.json file."
    )

class DatasetParams(Schema):
    """Dataset related parameters."""

    dataset_path: Path = Field(
        ..., 
        description="Path to the dataset root directory"
    )
    dataset_class: str = Field(
        ..., 
        description="Name of the dataset class to use"
    )
    split_strategy: str = Field(
        default="0.2", 
        description="How to split the data into train/val (passed to dataset creation)"
    )
    augmentation_strategy: str | None = Field(
        default=None, 
        description="Data augmentation procedure passed to dataset creation"
    )
    resize_size: tuple[int, int] = Field(
        default=(384, 384), 
        description="Size to resize the images. (Height, Width)"
    )
    # Replaces action=ParseKwargs
    dataset_params: dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters to pass to the dataset creation function."
    )
    loss_function: str = Field(
        default="cross_entropy", 
        description="Loss function to use during training"
    )
    ignore_class_weights: bool = Field(
        default=False, 
        description="If provided, ignore class weights for the loss function"
    )

    @field_validator("dataset_path")
    @classmethod
    def validate_path_exists(cls, v: Path) -> Path:
        # Optional: Uncomment if you want to enforce path existence at config load time
        # if not v.exists():
        #     raise ValueError(f"Dataset path does not exist: {v}")
        return v

class ModelParams(Schema):
    """Model related parameters."""

    model_class: str = Field(
        ..., 
        description="Name of the model to train"
    )
    weights_strategy: str | None = Field(
        default=None, 
        description="Defines how to load the weights"
    )
    model_params: dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional parameters to pass to the model creation function."
    )

class TrainingParams(Schema):
    """Training related parameters."""

    num_epochs: int = Field(
        default=2, ge=1, 
        description="Number of training epochs"
    )
    validation_metric: list[str] = Field(
        default=["Validation loss"], 
        description="Which metric to use for early stopping"
    )
    patience: int | None = Field(
        default=None, 
        description="Early stopping patience steps."
    )
    maximize_validation_metric: bool = Field(
        default=False, 
        description="If set, early stopping will maximize the metric instead of minimizing"
    )
    lr: float = Field(
        default=0.01, gt=0, 
        description="Initial learning rate"
    )
    lr_decay: float = Field(
        default=1.0, 
        description="Learning rate decay"
    )
    bs_train: int = Field(
        default=32, gt=0, 
        description="Batch size used during training"
    )
    bs_valid: int = Field(
        default=8, gt=0, 
        description="Batch size used during validation"
    )
    weight_decay: float = Field(
        default=1e-4, ge=0, 
        description="Weight decay for the optimizer"
    )
    optimizer: str = Field(
        default="sgd", 
        description="Optimizer to use"
    )
    momentum: float = Field(
        default=0.9, ge=0, 
        description="Momentum/beta1 of the optimizer"
    )
    seed: int = Field(
        default=0, 
        description="Seed for the random number generator"
    )

class DeviceParams(Schema):
    """Device and performance related parameters."""

    num_workers: int = Field(
        default=5, ge=0, 
        description="Number of workers for the DataLoader"
    )
    pin_memory: bool = Field(
        default=True, 
        description="If DataLoader should pin memory."
    )
    use_amp: bool = Field(
        default=False, 
        description="If automatic mixed precision should be used"
    )
    deterministic: bool = Field(
        default=False, 
        description="If deterministic algorithms should be used"
    )
    benchmark: bool = Field(
        default=False, 
        description="If cuda benchmark should be used"
    )
    profile: bool = Field(
        default=False, 
        description="If set, enable the profile mode."
    )
    profile_batches: int = Field(
        default=3, ge=1, 
        description="Number of batches to profile"
    )
    profile_verbosity: int = Field(
        default=0, 
        description="Profile verbosity (0, 1, or 2)"
    )

    @field_validator("device")
    @classmethod
    def validate_device_string(cls, v: str) -> str:
        valid_starts = ("cpu", "cuda", "mps")
        if not any(v.startswith(s) for s in valid_starts):
            raise ValueError(f"Device must start with one of {valid_starts}")
        return v
    
class ExperimentConfig(Schema):
    """Main configuration class that aggregates all subgroups."""

    logging: LoggingParams = Field(default_factory=LoggingParams)
    dataset: DatasetParams
    model: ModelParams
    training: TrainingParams = Field(default_factory=TrainingParams)
    device: DeviceParams = Field(default_factory=DeviceParams)
