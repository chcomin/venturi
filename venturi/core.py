"""Core classes for Venturi experiments."""

import gc
import importlib
import os
import shutil
import warnings
from pathlib import Path

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import Callback, DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from optuna.integration import PyTorchLightningPruningCallback

from venturi.config import Config, instantiate
from venturi.util import (
    ImageSaveCallback,  # For saving validation images
    LossCollection,  # For handling multiple losses
    OptunaConfigSampler,  # For Optuna hyperparameter sampling
    PlottingCallback,  # For plotting metrics
    TrainingTimeLoggerCallback,  # For logging training time
    delete_wandb_run,
    generate_name_from_config,
    get_next_name,
    is_rank_zero,
    patch_lightning,
    silence_lightning,
)

if importlib.util.find_spec("wandb") is None:
    _has_wandb = False
else:
    _has_wandb = True

patch_lightning()

torch.set_float32_matmul_precision("high")


class DataModule(pl.LightningDataModule):
    """Base DataModule which uses a dataset setup function defined in the config file."""

    def __init__(self, vcfg: Config):
        """Args:
        vcfg: Venturi configuration.
        """
        super().__init__()
        self.vcfg = vcfg

    def setup(self, stage=None):
        """Setup datasets for different stages.

        Args:
            stage: One of 'fit', 'validate' or 'test'.
        """

        vcfg_l = self.vcfg.logging
        # We need to silence lightning and wandb here due to multiprocessing
        if vcfg_l.silence_lightning:
            silence_lightning()
        if _has_wandb and vcfg_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        # dataloader generator
        self.generator = torch.Generator()
        if self.vcfg.seed is not None:
            self.generator.manual_seed(self.vcfg.seed)

        # Call the function indicated in self.vcfg.dataset.setup, passing vcfg.
        get_dataset = instantiate(self.vcfg.dataset.setup, partial=True)
        ds_dict = get_dataset(self.vcfg)
        self._check_datasets(stage, ds_dict)

        self.train_ds = ds_dict.get("train_ds", None)
        self.val_ds = ds_dict.get("val_ds", None)
        self.test_ds = ds_dict.get("test_ds", None)

    def train_dataloader(self):
        """Returns the training DataLoader."""
        vcfg_dl = self.vcfg.dataset.train_dataloader
        vcfg_dl["_target_"] = "torch.utils.data.DataLoader"
        dl = instantiate(vcfg_dl, partial=True)
        return dl(self.train_ds, generator=self.generator)

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        vcfg_dl = self.vcfg.dataset.val_dataloader
        vcfg_dl["_target_"] = "torch.utils.data.DataLoader"
        dl = instantiate(vcfg_dl, partial=True)
        return dl(self.val_ds, generator=self.generator)

    def test_dataloader(self):
        """Returns the test DataLoader."""
        vcfg_dl = self.vcfg.dataset.test_dataloader
        vcfg_dl["_target_"] = "torch.utils.data.DataLoader"
        dl = instantiate(vcfg_dl, partial=True)
        return dl(self.test_ds, generator=self.generator)

    def _check_datasets(self, stage, ds_dict):
        """Checks that the required datasets are present in ds_dict for the given stage."""

        expected_keys = {
            "fit": ["train_ds", "val_ds"],
            "validate": ["val_ds"],
            "test": ["test_ds"]
        }
        if stage in expected_keys:
            for key in expected_keys[stage]:
                if key not in ds_dict:
                    raise ValueError(
                        f"Dataset '{key}' is required for stage '{stage}' but not found.")

class TrainingModule(pl.LightningModule):
    """Base TrainingModule which uses model, loss and metric setup functions defined in the
    config file. The model is very close to a vanilla LightningModule.
    """

    def __init__(self, vcfg: Config):
        """Args:
        vcfg: Venturi configuration.
        """
        super().__init__()
        self.vcfg = vcfg

        get_model = instantiate(self.vcfg.model.setup, partial=True)
        self.pt_model = get_model(self.vcfg)

        normalize_weights = vcfg.losses.get("normalize_weights", False) 
        loss_fn = LossCollection(vcfg.losses, normalize_weights=normalize_weights)
        self.train_loss = loss_fn.clone(prefix="train/")
        self.val_loss = loss_fn.clone(prefix="val/")

        # Performance Metrics
        if "metrics" in self.vcfg:
            get_metrics = instantiate(self.vcfg.metrics.setup, partial=True)
            metrics = get_metrics(self.vcfg)
            self.val_metrics = metrics.clone()
            self.test_metrics = metrics.clone()
            if "preprocessing" in self.vcfg.metrics:
                self.preprocessing = instantiate(self.vcfg.metrics.preprocessing, partial=True)
            else:
                self.preprocessing = None
        else:
            self.val_metrics = None
            self.test_metrics = None

    def forward(self, x):
        """Forward pass through the model."""
        return self.pt_model(x)

    def training_step(self, batch):
        """Performs a training step."""
        x, y = batch
        logits = self(x)
        loss, loss_logs = self.train_loss(logits, y)

        bs = x.size(0)
        self.log("global_step", self.trainer.global_step)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        if len(loss_logs) > 1:
            self.log_dict(loss_logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

        return loss

    def validation_step(self, batch):
        """Performs a validation step."""
        x, y = batch
        logits = self(x)
        loss, loss_logs = self.val_loss(logits, y)

        bs = x.size(0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        if len(loss_logs) > 1:
            self.log_dict(loss_logs, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

        if self.val_metrics is not None:
            if self.preprocessing is not None:
                logits, y = self.preprocessing(logits, y) 
            self.val_metrics.update(logits, y)

        # Return logits for plotting callbacks
        return {"loss": loss, "logits": logits.detach()}
    
    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and reset metric state."""
        if self.val_metrics is not None:
            output = self.val_metrics.compute()
            output = {f"val/{k}": v for k, v in output.items()}  # Prefix keys with "val/"
            self.log_dict(output, prog_bar=False)
            self.val_metrics.reset()

    def test_step(self, batch):
        """Performs a test step."""
        x, y = batch
        logits = self(x)

        if self.test_metrics is not None:
            if self.preprocessing is not None:
                logits, y = self.preprocessing(logits, y)
            self.test_metrics.update(logits, y)

    def on_test_epoch_end(self) -> None:
        """Log test metrics and reset metric state."""
        if self.test_metrics is not None:
            output = self.test_metrics.compute()
            output = {f"test/{k}": v for k, v in output.items()}
            self.log_dict(output, prog_bar=False)
            self.test_metrics.reset()

    def configure_optimizers(self):
        """Configures optimizers and learning rate schedulers based on the config file."""

        vcfg_t = self.vcfg.training
        optimizer_factory = instantiate(vcfg_t.optimizer, partial=True)
        optimizer = optimizer_factory(self.parameters())

        output = {"optimizer": optimizer}

        if "lr_scheduler" in vcfg_t:
            output["lr_scheduler"] = self.get_scheduler(optimizer)

        return output

    def get_scheduler(self, optimizer):
        """This function just needs to return a lr_scheduler_config dictionary as described in
        the Lightning docs. The base function here implements a complicated logic to handle
        almost all Pytorch schedulers by just changing the yaml configuration.
        """

        vcfg_t = self.vcfg.training
        scheduler_factory = instantiate(vcfg_t.lr_scheduler.instance, partial=True)
        sched_args = {"optimizer": optimizer}

        # Some lr_schedulers need to know the total number of iterations
        if getattr(vcfg_t.lr_scheduler, "needs_total_iters", False):
            interval = getattr(vcfg_t.lr_scheduler.scheduler_config, "interval", "step")
            if interval == "epoch":
                total_iters = self.trainer.max_epochs
            else:
                total_iters = self._estimate_total_steps()
            if "OneCycleLR" in vcfg_t.lr_scheduler.instance:
                # In OneCycleLR the parameter is named total_steps instead of total_iters
                sched_args["total_steps"] = total_iters
            else:
                sched_args["total_iters"] = total_iters

        scheduler = scheduler_factory(**sched_args)

        lr_scheduler_config = vcfg_t.lr_scheduler.scheduler_config.to_dict()
        lr_scheduler_config["scheduler"] = scheduler

        return lr_scheduler_config

    def _estimate_total_steps(self):
        """Estimate total training steps for schedulers that need it."""

        try:
            total_iters = self.trainer.estimated_stepping_batches
        except Exception:
            total_iters = None

        if total_iters == float("inf") or (total_iters is None and self.trainer.max_epochs == -1):
            raise ValueError(
                "The selected scheduler requires a known total number of steps (total_iters), "
                "but `max_epochs` is set to -1 (infinite). Please set `max_epochs` to a positive "
                "integer or choose a different scheduler."
            )

        # Fallback calculation if Lightning returned None or 0 (but max_epochs is valid)
        if total_iters is None or total_iters == 0:
            if self.trainer.max_epochs > 0:  # type: ignore
                num_devices = max(1, self.trainer.num_devices)
                batch_size = self.vcfg.dataset.train_dataloader.batch_size
                dataset_len = len(self.trainer.datamodule.train_dataloader().dataset) # type: ignore

                factor = batch_size * num_devices * self.trainer.accumulate_grad_batches
                steps_per_epoch = dataset_len // factor
                total_iters = steps_per_epoch * self.trainer.max_epochs

                if total_iters == 0:
                    raise ValueError(
                        "Estimated total steps is 0. Check your batch size and dataset length."
                    )
            else:
                raise ValueError(
                    "Cannot estimate total steps. Ensure `max_epochs` > 0 or use `max_steps`."
                )

        return total_iters


class Experiment:
    """Main class to run experiments based on a configuration file."""

    def __init__(self, vcfg: Config):
        """Args:
        vcfg: Venturi configuration.
        """

        self._check_vcfg(vcfg)

        self.vcfg = vcfg
        self.run_path: Path | None = None
        self.model: TrainingModule | None = None
        self.trainer: pl.Trainer | None = None

        self._set_seed()

        self.data_module = self.get_data_module()

    def setup_logging(self) -> Path:
        """Handles the run_path name expansion and experiment directory creation logic as well
        as lightning and wandb verbosity.

        The method must return the main path where logs and checkpoints will be stored.

        Returns:
            Path: The path to the logging directory.
        """
        vcfg_l = self.vcfg.logging

        if vcfg_l.silence_lightning:
            silence_lightning()
        if _has_wandb and vcfg_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        # Expand run_path name if {key} patterns are found
        vcfg_copy = self.vcfg.copy()
        vcfg_copy.logging.run_path = ""
        run_path = generate_name_from_config(vcfg_copy, vcfg_l.run_path)
        run_path = Path(run_path)

        # Not fitting, no need to mess with folders, but run_path must exist. This should
        # be done only on rank 0, but doing here makes the code simpler and racing conditions
        # should be rare since fit has already created the folder.
        if self._stage != "fit":
            if not run_path.exists():
                raise ValueError(f"Run path {run_path} does not exist.")
            return run_path

        # The user does not want folder creation
        if not vcfg_l.create_folder:
            return run_path

        # Let rank 0 find a new name and create folders
        if is_rank_zero():
            # Resume from checkpoint, we just check if path exists and write a new config file
            if self.vcfg.training.resume_from_checkpoint is not None:
                if not run_path.exists():
                    raise ValueError(
                        f"Cannot resume from checkpoint, run_path {run_path} does not exist."
                    )
                # Rename previous config file to available config_prev_x.yaml
                current_vcfg_path = run_path / "config.yaml"
                prev_vcfg_path = run_path / "config_prev.yaml"
                prev_vcfg_path = get_next_name(prev_vcfg_path)
                current_vcfg_path.rename(prev_vcfg_path)
                self.vcfg.save(current_vcfg_path)
                return run_path

            if run_path.exists():
                if vcfg_l.overwrite_existing:
                    # Erase previous run
                    shutil.rmtree(run_path)
                else:
                    # Get next available run name
                    run_path = get_next_name(run_path)

            run_path.mkdir(parents=True, exist_ok=True)
            self.vcfg.save(run_path / "config.yaml")

        if torch.distributed.is_initialized():
            # Broadcast run_path to other processes
            path_container = [run_path] if is_rank_zero() else [None] # type: ignore
            torch.distributed.broadcast_object_list(path_container, src=0)
            run_path = path_container[0]
            torch.distributed.barrier()

        return run_path 

    def get_data_module(self) -> DataModule:
        """Override this for a different dataset logic (e.g.: multiple datasets)."""
        return DataModule(self.vcfg)

    def get_model(self) -> TrainingModule:
        """Override this for a different model logic (e.g.: multiple models)."""
        return TrainingModule(self.vcfg)

    def get_loggers(self) -> list[pl.loggers.Logger]:
        """Setup loggers. Loggers are used for writing logs to disk or to a remote service like
        Weights & Biases. 
        
        The method must return a list of loggers.
        
        Returns:
            list[pl.loggers.Logger]: The list of loggers.
        """

        vcfg_l = self.vcfg.logging
        run_path = self.run_path

        if run_path is None:
            raise ValueError("run_path must be set before setting loggers.")

        loggers: list[pl.loggers.Logger] = []

        if vcfg_l.log_csv:
            metrics_file = run_path / "metrics.csv"
            if metrics_file.exists() and is_rank_zero():
                # Condition only happens when resuming from checkpoint. Rename existing file.
                prev_metrics_file = run_path / "metrics_prev.csv"
                prev_metrics_file = get_next_name(prev_metrics_file)
                metrics_file.rename(prev_metrics_file)
            loggers.append(CSVLogger(save_dir=run_path, name="", version=""))

        vcfg_w = vcfg_l.wandb
        if vcfg_w.log_wandb:
            if _has_wandb is False:
                raise ImportError(
                    "WandbLogger requires wandb to be installed. "
                    "Please install it or disable wandb logging."
                )
            is_new_run = not self.vcfg.training.resume_from_checkpoint
            # If a new run and in fit stage, delete any existing wandb run with the same name
            if is_rank_zero() and is_new_run and self._stage == "fit":
                delete_wandb_run(vcfg_w.wandb_project, str(run_path))
            loggers.append(
                WandbLogger(
                    name=str(run_path),
                    save_dir=str(run_path),
                    project=vcfg_w.wandb_project,
                    group=vcfg_w.wandb_group,
                    config=self.vcfg.to_dict(),
                )
            )

        return loggers

    def get_callbacks(self, extra_callbacks: list[Callback] | None = None) -> list[Callback]:
        """Constructs a list of callbacks that are sent to Lightning. Callbacks can be used for
        model checkpointing, early stopping, custom validation logging, etc.
        
        The method must return a list of callbacks.

        Args:
        extra_callbacks: passed from get_trainer() (e.g. Optuna Pruning)

        Returns:
            list[Callback]: The list of callbacks.
        """

        if self.run_path is None:
            raise ValueError("run_path must be set before setting callbacks.")

        vcfg_l = self.vcfg.logging
        vcfg_t = self.vcfg.training
        callbacks = extra_callbacks or []

        # Device Stats Monitor
        if vcfg_t.monitor_device_stats:
            callbacks.append(DeviceStatsMonitor(cpu_stats=True))

        # Training Time Logger
        if vcfg_l.log_training_time:
            callbacks.append(TrainingTimeLoggerCallback(self.run_path))

        # Custom Visualization & Plotting
        if vcfg_l.save_val_data:
            callbacks.append(
                ImageSaveCallback(
                    self.run_path,
                    vcfg_l.val_data_indices,
                    log_disk=vcfg_l.log_val_data_to_disk,
                    mean=getattr(vcfg_l, "dataset_mean", None),
                    std=getattr(vcfg_l, "dataset_std", None),
                    log_wandb=vcfg_l.wandb.log_wandb and vcfg_l.wandb.log_val_data_to_wandb,
                    class_labels=vcfg_l.wandb.class_labels,
                )
            )

        if vcfg_l.log_plot:
            callbacks.append(PlottingCallback())

        mode = "max" if vcfg_t.maximize_validation_metric else "min"

        # Checkpointing
        if vcfg_l.log_checkpoints:
            # Best Model
            if vcfg_l.save_top_k_models > 0:
                val_m = vcfg_t.validation_metric
                safe_val_m = val_m.replace("/", "_")
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=self.run_path / "models",
                        filename=f"best_model_epoch={{epoch}}_{safe_val_m}={{{val_m}:.4f}}",
                        save_top_k=vcfg_l.save_top_k_models,
                        save_last=False,
                        monitor=val_m,
                        mode=mode,
                        auto_insert_metric_name=False,
                    )
                )
            # Save model every n epochs
            if vcfg_l.save_model_every_n_epochs > 0:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=self.run_path / "models",
                        filename="last",
                        save_top_k=1,
                        save_last=False,
                        monitor=None,
                        every_n_epochs=vcfg_l.save_model_every_n_epochs,
                    )
                )

        # Early Stopping
        if vcfg_t.patience:
            callbacks.append(
                EarlyStopping(
                    monitor=vcfg_t.validation_metric,
                    patience=vcfg_t.patience,
                    mode=mode,
                    divergence_threshold=vcfg_t.divergence_threshold,
                )
            )

        # Instantiate extra callbacks from config
        if "extra_callbacks" in vcfg_t:
            for cb_conf in vcfg_t.extra_callbacks.values():
                cb = instantiate(cb_conf)
                callbacks.append(cb)

        return callbacks

    def get_profiler(self) -> PyTorchProfiler | None:
        """Setup a profiler for measuring performance.
        
        Returns:
            PyTorchProfiler | None: The profiler or None if profiling is disabled.
        """

        if not getattr(self.vcfg.training, "profile", False):
            return None

        verbosity = getattr(self.vcfg.training, "profile_verbosity", 0)

        if verbosity >= 5:
            # TODO: check if this works
            experimental_config = torch.profiler._ExperimentalConfig(  
                verbose=True,
                profiler_metrics=[
                    "kineto__cuda_core_occupancy",  # GPU occupancy
                    "kineto__dram_throughput",  # GPU memory bandwidth
                ],
            )
        else:
            experimental_config = None

        profiler = PyTorchProfiler(
            dirpath=str(self.run_path),
            filename="profile",
            export_to_chrome=True,
            with_flops=verbosity >= 1,
            record_shapes=verbosity >= 2,
            profile_memory=verbosity >= 3,
            with_stack=verbosity >= 4,
            experimental_config=experimental_config,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=4, repeat=1),
        )

        return profiler

    def get_trainer(self, extra_callbacks: list[Callback] | None = None) -> pl.Trainer:
        """Sets up the PyTorch Lightning Trainer. Must return a pl.Trainer instance.
        
        Args:
            extra_callbacks: Extra callbacks to add to the trainer.
        
        Returns:
            pl.Trainer: The PyTorch Lightning Trainer.
        """

        loggers = self.get_loggers()
        callbacks = self.get_callbacks(extra_callbacks)
        profiler = self.get_profiler()

        enable_checkpointing = any(isinstance(cb, ModelCheckpoint) for cb in callbacks)

        vcfg_t = self.vcfg.training.trainer_params

        trainer = pl.Trainer(
            default_root_dir=str(self.run_path),
            devices=vcfg_t.devices,
            strategy=vcfg_t.strategy,
            precision=vcfg_t.precision,
            max_epochs=vcfg_t.max_epochs,
            accumulate_grad_batches=vcfg_t.accumulate_grad_batches,
            gradient_clip_val=vcfg_t.gradient_clip_val,
            check_val_every_n_epoch=vcfg_t.check_val_every_n_epoch,
            log_every_n_steps=self.vcfg.logging.log_every_n_steps,
            enable_progress_bar=self.vcfg.logging.enable_progress_bar,
            deterministic=vcfg_t.deterministic,
            benchmark=vcfg_t.benchmark,
            enable_checkpointing=enable_checkpointing,
            logger=loggers,
            callbacks=callbacks,
            profiler=profiler,
            enable_model_summary=False,
            **vcfg_t.kwargs,
        )

        return trainer

    def fit(
        self,
        vcfg_overrides: Config | dict | None = None,
        extra_callbacks: list[Callback] | None = None,
        recreate_dataset: bool = False,
    ):
        """Main entry point for training.

        Args:
            vcfg_overrides: Dictionary to override vcfg in self.vcfg.
            extra_callbacks: Extra callbacks to add to the trainer.
            recreate_dataset: If True, recreates the data module. Useful for hyperparameter
            optimization where dataset parameters may change.

        Returns:
            float: The value of the validation metric after training (useful for hyperparameter
            optimization).
        """

        self._stage = "fit"

        gc.collect()
        torch.cuda.empty_cache()

        if vcfg_overrides is not None:
            self.vcfg.update_from(vcfg_overrides)

        self._set_seed()

        # Recreate dataset if needed
        if recreate_dataset or self.data_module is None:
            self.data_module = self.get_data_module()

        self.run_path = self.setup_logging()
        self.model = self.get_model()
        self.trainer = self.get_trainer(extra_callbacks=extra_callbacks)
        # Set checkpoint path if resuming
        ckpt_path = None
        ckpt_name = self.vcfg.training.resume_from_checkpoint
        if ckpt_name is not None:
            ckpt_path = self.run_path / "models" / ckpt_name

        # Train!
        self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path=ckpt_path)

        # Return metric (Useful for Optuna)
        val_m = self.vcfg.training.validation_metric
        if val_m in self.trainer.callback_metrics:
            final_metric = self.trainer.callback_metrics[val_m].item()
        else:
            raise ValueError(
                f"Validation metric {val_m} not found in trainer metrics. "
                "Is the name correct, and did fit actually run?"
                )

        return final_metric

    def test(
        self,
        vcfg_overrides: Config | dict | None = None,
        checkpoint_name: str | None = None,
        recreate_dataset: bool = False,
    ):
        """Test a model. The model, trainer and, optionally, data module are recreated at
        the start of testing.

        This function may be called on two different scenarios:

        1) Right after fit(): the default behavior is to use the best model found during fit if
        log_checkpoints was True during training. This can be changed by passing a specific
        checkpoint name or by setting checkpoint="last" to use the last saved checkpoint.
        2) Without calling fit(): test can be called for any pretrained model by setting the
        appropriate run_path in the config file and passing a specific checkpoint name.

        Args:
            vcfg_overrides: Dictionary to override vcfg in self.vcfg. Be careful when changing
            parameters here, since they must be compatible with the training parameters. You can
            change, for instance, the batch size.
            checkpoint_name: Name of the checkpoint to load from self.run_path/models or "last".
            See description above for details.
            recreate_dataset: If True, recreates the data module.
        """

        self._stage = "test"

        gc.collect()
        torch.cuda.empty_cache()

        if vcfg_overrides is not None:
            self.vcfg.update_from(vcfg_overrides)

        fit_called = self.run_path is not None

        if not fit_called and (checkpoint_name is None or checkpoint_name == "last"):
            raise ValueError(
                "You must provide a checkpoint name when calling test() without calling fit() "
                "beforehand."
            )

        self._set_seed()

        if recreate_dataset or self.data_module is None:
            self.data_module = self.get_data_module()

        if not fit_called:
            self.run_path = self.setup_logging()
            self.model = self.get_model()
            self.trainer = self.get_trainer()

        # Sanity check to make sure all attributes are set
        for attr in (self.run_path, self.data_module, self.model, self.trainer):
            if attr is None:
                raise ValueError(f"Attribute {attr} is None. Cannot proceed with testing.")

        # Checkpoint loading logic
        ckpt_path = None
        if checkpoint_name is None:
            if self.trainer and self.trainer.checkpoint_callback:
                # Warning! This relies on the save_top_k checkpoint callback being present and
                # being the first ModelCheckpoint on the list of callbacks!
                ckpt_path = self.trainer.checkpoint_callback.best_model_path # type: ignore
            if not ckpt_path or ckpt_path == "":
                raise ValueError(
                    "Could not find best model path from trainer. Please provide a checkpoint name."
                )
        elif checkpoint_name == "last":
            ckpt_path = self.run_path / "models" / "last.ckpt"  # type: ignore
        else:
            ckpt_path = self.run_path / "models" / checkpoint_name  # type: ignore
        # ----

        return self.trainer.test(  # type: ignore
            self.model, datamodule=self.data_module, ckpt_path=ckpt_path
        )
    
    def optimize(self, vcfg_space, vcfg_optuna: Config | None = None):
        """Optimize hyperparameters using Optuna.

        Args:
            vcfg_space: Configuration defining the hyperparameter search space.
            vcfg_optuna: Configuration defining the Optuna study parameters. If None, uses
            self.vcfg.optuna.
        """

        if vcfg_optuna is None:
            if "optuna" not in self.vcfg:
                raise ValueError("No Optuna study configuration provided.")
            vcfg_optuna = self.vcfg.optuna

        if getattr(vcfg_optuna, "silence_optuna", False):
            warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Optuna objective function. This function is here to not pollute the class namespace.
        # This is not safe for pickling in case of distributed optimization with spawned processes,
        # but it is unlikely that .optimize() will be called in spawned processes.
        def objective(trial: optuna.Trial):
            gc.collect()
            torch.cuda.empty_cache()
            
            # Sample hyperparameters
            overrides = optuna_config_sampler.sample(trial, include_category=False)
            trial.set_user_attr("vcfg_overrides", overrides.to_dict())

            pruning_cb = PyTorchLightningPruningCallback(
                trial, monitor=self.vcfg.training.validation_metric)
            
            return self.fit(overrides, extra_callbacks=[pruning_cb])  

        optuna_config_sampler = OptunaConfigSampler(vcfg_space)

        # Create Optuna study
        vcfg_os = instantiate(vcfg_optuna.study)
        study = optuna.create_study(**vcfg_os)
        
        # Optimize, passing the experiment and search space to the objective function
        study.optimize(objective, **vcfg_optuna.study_optimize)
        
        return study  

    def _set_seed(self):
        if self.vcfg.seed is not None:
            pl.seed_everything(self.vcfg.seed, workers=True, verbose=False)

    def _check_vcfg(self, vcfg: Config):
        """Do some checks on vcfg to avoid misconfigurations."""

        if not is_rank_zero():
            return

        if vcfg.dataset.setup._target_ == "<dot.path.to.function>":
            raise ValueError("Dataset setup function is not defined in the configuration.")
        if vcfg.model.setup._target_ == "<dot.path.to.function>":
            raise ValueError("Model setup function is not defined in the configuration.")
        if vcfg.metrics.setup._target_ == "<dot.path.to.function>":
            raise ValueError("Metrics function is not defined in the configuration.")
        if len(vcfg.losses) == 0:
            raise ValueError("No losses defined in the 'losses' section of the configuration.")
        has_weight = ["loss_weight" in param for param in vcfg.losses.values()]
        if len(vcfg.losses) > 1 and not all(has_weight):
            raise ValueError(
                "When using more than one loss, all losses in the 'losses' section must have a "
                "'loss_weight' parameter."
            )

        vcfg_l = vcfg.logging
        has_log = any(
            [
                vcfg_l.log_csv,
                vcfg_l.log_checkpoints,
                vcfg_l.log_plot,
                vcfg_l.log_training_time,
                vcfg_l.save_val_data,
                # Decided to not include wandb here since any logging to wandb requires folder
                # creation. It may happen that a user does not want any local logging, but wants to
                # log to wandb. This cannot be done without creating a local folder.
            ]
        )
        if has_log and not vcfg_l.create_folder:
            raise ValueError("You have a logger enabled but create_folder is False.")

        log_val = vcfg_l.log_val_data_to_disk or vcfg_l.wandb.log_val_data_to_wandb
        if vcfg_l.save_val_data and not log_val:
            raise ValueError("You need to log to disk or wandb if save_val_data is True.")
