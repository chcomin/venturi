import contextlib
import logging
import os
import string
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import optuna
import pandas as pd
import torch
from lightning.fabric.loggers import csv_logs
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from PIL import Image
from torch import nn
from torchmetrics import Metric, MetricCollection

from venturi.config import Config, instantiate

try:
    import wandb
except ImportError:
    _has_wandb = False
else:
    _has_wandb = True

if os.environ.get("DISPLAY", "") == "" and "ipykernel" not in sys.modules:
    # No display and not using a notebook. Use non-interactive Agg backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LossCollection(nn.Module):
    """Container class to combine multiple loss functions.

    The constructor takes a Config object with the format:

    loss1_name:
        instance:
            _target_: path.to.LossClass
            arg1: value1
            ...
        loss_weight: float
    loss2_name:
        ...

    Each call to the forward method generates the total loss and, if return_logs is True, a
    dictionary with the individual loss values with the format {loss_name: loss_value}.
    """

    def __init__(
            self, 
            vcfg_loss: Config, 
            return_logs: bool = True, 
            normalize_weights: bool = False,
            prefix: str = ""
            ):
        """Args:
        vcfg_loss: Configuration dictionary for the loss functions.
        return_logs: Whether to return individual loss values as logs.
        normalize_weights: Whether to normalize the weights to sum to 1.0.
        prefix: Prefix to add to the loss names in the logs.
        """
        super().__init__()
        

        # If there is a single loss without weight, set its weight to 1.0
        if len(vcfg_loss) == 1:
            name, config = next(iter(vcfg_loss.items()))
            if "loss_weight" not in config:
                config["loss_weight"] = 1.0

        self.loss_map = nn.ModuleDict()
        self.vcfg_loss = vcfg_loss
        self.return_logs = return_logs
        self.normalize_weights = normalize_weights
        self.weights = {}

        # Register components
        for name, config in vcfg_loss.items():
            self.loss_map[f"{prefix}{name}"] = instantiate(config["instance"])
            self.weights[f"{prefix}{name}"] = config["loss_weight"]

        if normalize_weights:
            total_weight = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] /= total_weight

    def forward(self, input, target):
        """Passes the input and target to every child loss.

        Args:
            input: Output from the network.
            target: Ground truth data.
        """
        total_loss = 0.0
        logs = {}

        for name, loss_fn in self.loss_map.items():
            weight = self.weights[name]

            val = loss_fn(input, target)

            total_loss += weight * val
            logs[name] = val.detach()

        output = (total_loss, logs) if self.return_logs else total_loss

        return output

    def clone(self, prefix: str = "") -> "LossCollection":
        """Make a copy of the class."""
        return self.__class__(
            deepcopy(self.vcfg_loss), self.return_logs, self.normalize_weights, prefix)


class MetricCollectionWrapper(nn.Module):
    """A base class that wraps a MetricCollection and allows for custom 
    preprocessing of inputs before they reach the metrics.
    """
    def __init__(
        self, 
        metrics: MetricCollection | dict[str, Metric], 
        prefix: str | None = None
    ):
        super().__init__()
        if isinstance(metrics, dict):
            metrics = MetricCollection(metrics)
        
        if prefix:
            metrics = metrics.clone(prefix=prefix)
            
        self.collection = metrics

    def preprocess(self, logits: torch.Tensor, target: torch.Tensor):
        """Override this method in subclasses to preprocess the input data.
        
        Args:
            logits: Output from the network.
            target: Ground truth data.

        Returns:
            A tuple (preds, new_target) after preprocessing.
        """
        return logits, target

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        """Preprocesses data and updates the internal collection."""
        preds, target = self.preprocess(logits, target)
        self.collection.update(preds, target)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        self.update(logits, target)

    def compute(self) -> dict[str, torch.Tensor]:
        return self.collection.compute()

    def reset(self):
        self.collection.reset()

    def clone(self, prefix: str | None = None):
        """Creates a copy of this class."""
        new_collection = self.collection.clone(prefix=prefix)
        
        new_instance = self.__class__(metrics=new_collection)
        return new_instance

    def __getitem__(self, key: str) -> Metric:
        return self.collection[key]

    def items(self):
        return self.collection.items()
    
    def keys(self):
        return self.collection.keys()
        
    def values(self):
        return self.collection.values()

    def __len__(self):
        return len(self.collection)

class OptunaConfigSampler:
    """Parses a hierarchical search space configuration and samples values 
    using an Optuna trial.

    The search space configuration is defined as a nested dictionary where
    hyperparameters can be specified with distributions. For instance:

    params:
        lr:
            type: float
            low: 1e-4
            high: 1e-2
            log: true
        momentum:
            type: float
            low: 0.5
            high: 0.99

    The class also supports branching on categorical choices. For example:

    model:
        type: categorical
        choices: [cnn, transformer]
        cnn:
            setup:
                _target_: models.get_cnn
            num_layers:
                type: int
                low: 2
                high: 10
        transformer:
            setup:
                _target_: models.get_vit
            num_heads:
                type: int
                low: 2
                high: 8

    If necessary, branching can also be done on integer indices mapping to lists of options:

    training:
        optimizer:
            type: int
            low: 0
            high: 1
            _options_:
                - sgd:
                    lr:
                        type: float
                        low: 1e-4
                        high: 1e-2
                - adam:
                    lr:
                        type: float
                        low: 1e-5
                        high: 1e-3
    """
    
    def __init__(self, search_space: Config | dict):
        """Args:
        search_space: A configuration (Config or dict) defining the search space.
        """
        if hasattr(search_space, "to_dict"):
            search_space = search_space.to_dict()
        self.search_space = search_space

    def sample(self, trial: optuna.Trial, include_category: bool = True) -> Config:
        """Sample parameters from the search space.

        When branching on categorical choices, if `include_category` is True,
        the sampled parameters will include the category name in the dot-notation path.
        For instance, instead of:
            "parent.params.value"
        it becomes:
            "parent.choice_name.params.value"

        Args:
            trial: The Optuna trial object.
            include_category: If True, appends the categorical choice to the parameter name.

        Returns:
            A Config object with sampled parameters.
        """

        output = self._recurse(
            self.search_space, trial, prefix="", include_category=include_category)

        return Config(output)

    def _recurse(self, node: Any, trial: optuna.Trial, prefix: str, include_category: bool) -> Any:
        # Base case: Not a dict
        if not isinstance(node, dict):
            return node

        # Check for hyperparameter definition
        if "type" in node and node["type"] in ["int", "float", "categorical"]:
            return self._sample_distribution(node, trial, prefix, include_category)

        # Recursive case: Standard container
        sampled_dict = {}
        for key, value in node.items():
            # Build dot-notation path
            new_prefix = f"{prefix}.{key}" if prefix else key
            sampled_dict[key] = self._recurse(value, trial, new_prefix, include_category)
            
        return sampled_dict

    def _sample_distribution(
            self, node: dict, trial: optuna.Trial, name: str, include_category: bool) -> Any:
        dist_type = node["type"]

        if dist_type == "categorical":
            choices = node["choices"]
            choice_key = trial.suggest_categorical(name, choices)

            if choice_key in node:
                branch_config = node[choice_key]
                
                if include_category:
                    next_prefix = f"{name}.{choice_key}"
                else:
                    next_prefix = name
                
                sampled_branch = self._recurse(branch_config, trial, next_prefix, include_category)
                
                if isinstance(sampled_branch, dict):
                    sampled_branch["_replace_"] = True
                    
                return sampled_branch
            
            return choice_key
        
        elif dist_type == "int":
            val = trial.suggest_int(
                name, node["low"], node["high"], 
                step=node.get("step", 1), log=node.get("log", False)
            )

            # Check if the integer maps to a list of options (Branching)
            if "_options_" in node:
                options_list = node["_options_"]
                
                if val < 0 or val >= len(options_list):
                    raise ValueError(
                        f"Sampled index {val} is out of bounds for options list at '{name}'")
                
                selected_branch = options_list[val]
                
                # Strategy: If the branch is a dict with exactly ONE key (e.g. {'cnn': ...}), 
                # we treat that key as the branch name (better for readability/Optuna paths).
                # Otherwise, use the integer index as the name.
                if isinstance(selected_branch, dict) and len(selected_branch) == 1:
                    branch_name = next(iter(selected_branch.keys()))
                    branch_content = selected_branch[branch_name]
                else:
                    branch_name = str(val)
                    branch_content = selected_branch

                if include_category:
                    next_prefix = f"{name}.{branch_name}"
                else:
                    next_prefix = name

                sampled_content = self._recurse(
                    branch_content, trial, next_prefix, include_category)
                
                # Inject Replacement Flag
                if isinstance(sampled_content, dict):
                    sampled_content["_replace_"] = True
                
                return sampled_content

            return val

        elif dist_type == "float":
            return trial.suggest_float(
                name, node["low"], node["high"], 
                step=node.get("step"), log=node.get("log", False)
            )
        
        else:
            raise ValueError(f"Unknown distribution type '{dist_type}' at {name}")


class PlottingCallback(Callback):
    """Reads the CSV log generated by Lightning and creates a static plot (plots.png) at the
    end of every training epoch.
    """

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.log_dir is None:
            raise ValueError("Trainer log_dir is None; cannot save plots.")
        logger_dir = Path(trainer.log_dir)
        metrics_file = logger_dir / "metrics.csv"

        if not metrics_file.exists():
            raise ValueError(
                f"Metrics file not found: {metrics_file}. Maybe you forgot to enable CSVLogger?"
            )

        try:
            try:
                df = pd.read_csv(metrics_file)
            except Exception:
                # There might be racing conditions in multi-GPU training. If reading fails,
                # we skip plotting for this epoch.
                return

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))

            # Loss Plot
            vcfg_p = pl_module.vcfg.logging.plot.left_plot
            has_line = False
            for key in vcfg_p.metrics:  
                key_p = key
                if key_p == "train/loss":
                    # Lightning automatically adds 'step' or 'epoch' suffixes to train loss
                    key_p = "train/loss_epoch"
                if key_p in df.columns:
                    data = df[["epoch", key_p]].dropna()
                    if len(data) > 1:
                        ax[0].plot(data["epoch"], data[key_p], label=key)
                        has_line = True

            ax[0].set_xlabel("epoch")
            if has_line:
                ax[0].legend()
            ax[0].set_ylim(vcfg_p.ylim.min, vcfg_p.ylim.max)

            # Metrics Plot
            vcfg_p = pl_module.vcfg.logging.plot.right_plot
            has_line = False
            for key in vcfg_p.metrics:  
                if key in df.columns:
                    data = df[["epoch", key]].dropna()
                    if len(data) > 1:
                        ax[1].plot(data["epoch"], data[key], label=key)
                        has_line = True

            ax[1].set_xlabel("epoch")
            if has_line:
                ax[1].legend()
            ax[1].set_ylim(vcfg_p.ylim.min, vcfg_p.ylim.max)
            # Save
            plt.tight_layout()
            plt.savefig(logger_dir / "plots.png")
            plt.close(fig)

        except Exception as e:
            print(f"Error generating plot: {e}")


class ImageSaveCallback(Callback):
    """Saves validation predictions to disk."""

    def __init__(
        self,
        run_path: str | Path,
        val_img_indices: list[int],
        log_disk: bool = True,
        mean: torch.Tensor | float | None = None,
        std: torch.Tensor | float | None = None,
        log_wandb: bool = False,
        class_labels: list[str] | None = None,
    ):
        """Args:
        run_path: Path to the run folder where images will be saved.
        val_img_indices: List of image indices to log.
        log_disk: Whether to save images to disk.
        mean: Mean used for normalization (for denormalizing images).
        std: Std used for normalization (for denormalizing images).
        log_wandb: Whether to also log images to wandb.
        class_labels: List of class labels for segmentation masks. Only used by wandb logging.
        """
        super().__init__()

        run_path = Path(run_path)

        self.run_path = run_path
        self.val_img_indices = set(val_img_indices)
        self.log_disk = log_disk
        self.log_wandb = log_wandb
        self.mean = mean
        self.std = std
        self.class_labels = class_labels

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.log_disk:
            # Create folders
            for idx in self.val_img_indices:
                save_dir = self.run_path / "images" / f"image_{idx}"
                save_dir.mkdir(parents=True, exist_ok=True)

    # Only the main process logs images in multi-GPU training, which will miss some
    # indices but avoids complications.
    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0 
    ):
        inputs, targets = batch
        batch_size = inputs.size(0)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        # Check if any target index is in this batch
        current_indices = set(range(start_idx, end_idx))
        intersect = self.val_img_indices.intersection(current_indices)

        if not intersect:
            return

        with torch.no_grad():
            logits = outputs["logits"]

            # Apply argmax for multiclass or sigmoid>0.5 for binary
            if logits.shape[1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1).unsqueeze(1).float()

        for global_idx in intersect:
            local_idx = global_idx - start_idx

            img_tensor = inputs[local_idx].cpu()
            target_tensor = targets[local_idx].cpu()
            pred_tensor = preds[local_idx].cpu()

            img_prep = img_tensor

            # Denormalize image if necessary
            if self.mean is not None and self.std is not None:
                mean = torch.as_tensor(self.mean, device=img_prep.device)
                std = torch.as_tensor(self.std, device=img_prep.device)
                if mean.ndim == 0:
                    mean = mean.view(1, 1, 1)
                else:
                    mean = mean.view(-1, 1, 1)

                if std.ndim == 0:
                    std = std.view(1, 1, 1)
                else:
                    std = std.view(-1, 1, 1)
                if img_prep.shape[0] == 1 and (mean.shape[0] > 1 or std.shape[0] > 1):
                    raise ValueError("Image has 1 channel but mean and std have multiple values.")

                img_prep = img_prep * std + mean

            # Convert to HWC and scale to [0,255]
            if img_prep.ndim == 3 and img_prep.shape[0] in [1, 3, 4]:
                img_prep = img_prep.permute(1, 2, 0).squeeze()  # C,H,W -> H,W,C

            if img_prep.min() >= 0.0 and img_prep.max() <= 1.0:
                img_prep = img_prep * 255.0
            elif img_prep.min() >= -1.0 and img_prep.max() <= 1.0:
                img_prep = (img_prep + 1.0) * 127.5
            elif img_prep.min() < 0.0 or img_prep.max() > 255:
                min = img_prep.min()
                img_prep = (img_prep - min) / (img_prep.max() - min) * 255.0
            img_prep = img_prep.clamp(0, 255)
            target_tensor = target_tensor.squeeze()
            pred_tensor = pred_tensor.squeeze()

            if self.log_disk:
                self._save_to_disk(trainer, global_idx, img_prep, target_tensor, pred_tensor)
            if self.log_wandb:
                self._log_to_wandb(trainer, global_idx, img_prep, target_tensor, pred_tensor)

    def _save_to_disk(self, trainer, idx, img, target, pred):
        # Scale target and pred to [0,255]
        if target.max() > 0:
            target = target.float()
            target = target / target.max() * 255
        if pred.max() > 0:
            pred = pred.float()
            pred = pred / pred.max() * 255
        target = target.byte().numpy()
        pred = pred.byte().numpy()
        img = img.byte().numpy()

        img_pil = Image.fromarray(img)
        target_pil = Image.fromarray(target)
        pred_pil = Image.fromarray(pred)
        # Create image with 3 panels: input, target, prediction
        w, h = img_pil.size
        combined = Image.new("RGB", (w * 3, h))
        combined.paste(img_pil.convert("RGB"), (0, 0))
        combined.paste(target_pil.convert("RGB"), (w, 0))
        combined.paste(pred_pil.convert("RGB"), (w * 2, 0))

        filename = f"epoch_{trainer.current_epoch}.png"
        combined.save(self.run_path / "images" / f"image_{idx}" / filename)

    def _log_to_wandb(self, trainer, idx, img, target, pred):
        """Logs an interactive segmentation mask."""

        if not _has_wandb:
            raise ImportError("wandb is not installed. Cannot log images to wandb.")

        masks = {
            "predictions": {"mask_data": pred.byte().numpy()},
            "ground_truth": {"mask_data": target.byte().numpy()},
        }
        class_labels = self.class_labels
        if class_labels is not None:
            class_labels_dict = dict(enumerate(class_labels))
            masks["predictions"]["class_labels"] = class_labels_dict
            masks["ground_truth"]["class_labels"] = class_labels_dict

        # Create the WandB Image object
        wandb_image = wandb.Image(
            img.byte().numpy(),
            masks=masks,
            caption=f"Image {idx}",
        )

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(
                    {f"val_predictions/image_{idx}": [wandb_image]}, step=trainer.global_step
                )


class TimeLoggerCallback(Callback):
    """Logs the time taken for each training and validation epoch."""

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_start = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.train_epoch_start
        pl_module.log("train/epoch_time", epoch_time, on_epoch=True)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_epoch_start = time.time()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch_time = time.time() - self.val_epoch_start
        pl_module.log("val/epoch_time", epoch_time, on_epoch=True)


class TrainingTimeLoggerCallback(Callback):
    """Logs the time taken for a complete training run."""

    def __init__(self, run_path: str | Path):
        super().__init__()
        self.run_path = Path(run_path)

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.run_path / "training_time.txt", "w") as f:
            f.write(f"Training started at: {dt_string}\n")

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.run_path / "training_time.txt", "a") as f:
            f.write(f"Training ended at: {dt_string}\n")


class LightningLogFilter(logging.Filter):
    """Filter for annoying Lightning messages."""

    def filter(self, record):
        msg = str(record.msg)

        forbidden_phrases = [
            "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES",
        ]

        # Verify if any forbidden phrase is in the message
        return not any(phrase in msg for phrase in forbidden_phrases)


def is_rank_zero():
    """Check if the current process is rank zero."""
    return int(os.environ.get("RANK", "0")) == 0


def silence_lightning():
    """Silence annoying Lightning messages."""

    warnings.filterwarnings("ignore", module="lightning")
    # Warning when num_workers is low in the Dataloader
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    # Warning when logging/validation steps is larger than the number of batches
    warnings.filterwarnings("ignore", ".*The number of training batches.*")
    # Warning when ModelCheckpoint directory exists
    warnings.filterwarnings("ignore", ".*exists and is not empty.*")
    # Specific internal Lightning deprecation warning that might be solved in future versions
    warnings.filterwarnings("ignore", ".*treespec, LeafSpec.*")
    warnings.filterwarnings(
        "ignore", message=".*isinstance.treespec, LeafSpec. is deprecated.*", category=FutureWarning
    )

    lightning_filter = LightningLogFilter()

    # Apply the filter to the root of Lightning
    logging.getLogger("lightning.pytorch").addFilter(lightning_filter)
    logging.getLogger("lightning.fabric").addFilter(lightning_filter)

    # The above is not enough, so we iterate over all active loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if "lightning" in logger_name:
            logger = logging.getLogger(logger_name)
            if isinstance(logger, logging.Logger):
                logger.addFilter(lightning_filter)
                logger.setLevel(logging.ERROR)

def _noop(self): pass

def patch_lightning():
    """Monkey patches for lightning."""
    # Avoid Lightning erasing the log directory
    csv_logs._ExperimentWriter._check_log_dir_exists = _noop # type: ignore


def find_key_recursive(data: Config, target_key: str) -> Any:
    """Recursively searches for a key in a nested dictionary.
    Returns the value of the first match found.
    """
    if target_key in data:
        return data[target_key]

    for value in data.values():
        if isinstance(value, Config):
            result = find_key_recursive(value, target_key)
            if result is not None:
                return result

    return None


def generate_name_from_config(config: Config, template: str) -> str:
    """Fills a template string using values found recursively in a Config.

    Args:
        config: The configuration object.
        template: A string like "model_{arch}_lr{lr:.4f}".

    Returns:
        The formatted string.

    Raises:
        KeyError: If a key required by the template is not found in the config.
    """
    # Extract all variable names from the template (e.g., "lr", "bs")
    # Formatter.parse returns tuples: (literal_text, field_name, format_spec, conversion)
    needed_keys = {
        field_name for _, field_name, _, _ in string.Formatter().parse(template) if field_name
    }

    # Find values for these keys
    context = {}
    for key in needed_keys:
        value = find_key_recursive(config, key)

        if value is None:
            raise KeyError(f"Template requires key '{key}', but it was not found in the config.")

        context[key] = value

    # Format the string using the found values
    return template.format(**context)


def delete_wandb_run(project_name: str, run_path: str):
    """Delete a wandb run."""

    api = wandb.Api()
    runs = api.runs(project_name, {"config.logging.run_path": run_path})
    with contextlib.suppress(ValueError, TypeError):
        # Get number of runs to verify if there are any
        len(runs)

    for run in runs:
        run.delete()


def get_next_name(path: Path) -> Path:
    """If 'path' exists, appends _2, _3, etc. until a free name is found."""

    directory = path.parent
    base_name = path.stem
    if not path.exists():
        return path

    counter = 2
    while True:
        candidate = directory / f"{base_name}_{counter}{path.suffix}"
        if not candidate.exists():
            return candidate
        counter += 1

def setup_storage(storage_input: str | None) -> str | None:
    """Parses an Optuna storage string (SQLite), creates the parent directory
    if it doesn't exist, and returns the valid storage URL.

    Args:
        storage_input: A string like "sqlite:///data/db.sqlite3" or "data/db.sqlite3".
    """

    if storage_input is None:
        return None
    
    prefix = "sqlite:///"
    
    # Check if the user passed a full URL or just a path
    if storage_input.startswith(prefix):
        # Strip the prefix to get the actual file path
        file_path_str = storage_input[len(prefix):]
    else:
        file_path_str = storage_input
    
    # Convert to a Path object
    db_path = Path(file_path_str)

    if db_path.parent:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
    if storage_input.startswith(prefix):
        return storage_input
    else:
        return f"{prefix}{file_path_str}"