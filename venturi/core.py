"""Core classes for Venturi experiments."""

import importlib
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader

from venturi._util import (
    ImageSaveCallback,
    LossCollection,
    PlottingCallback,
    TrainingTimeLoggerCallback,
    delete_wandb_run,
    generate_name_from_config,
    get_next_experiment_name,
    silence_lightning,
)
from venturi.config import Config, instantiate

if importlib.util.find_spec("wandb") is None:
    _has_wandb = False
else:
    _has_wandb = True

torch.set_float32_matmul_precision("high")

class DataModule(pl.LightningDataModule):
    """Base DataModule which uses a dataset setup function defined in the config file."""
    def __init__(self, args: Config):
        """Args:
        args: Configuration dictionary.
        """
        super().__init__()
        self.args = args
        # Dataset dictionary to hold train, val, test or predict datasets
        self.ds_dict: dict[str, torch.utils.data.Dataset] = {}

    def setup(self, stage=None):
        """Setup datasets for different stages: 'fit', 'validate', 'test', or 'predict'."""

        args_l = self.args.logging
        # We need to silence lightning and wandb here due to multiprocessing
        if args_l.silence_lightning:
            silence_lightning()
        if _has_wandb and args_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        # Call the function indicated in self.args.dataset.setup, passing the args.
        get_dataset = instantiate(self.args.dataset.setup, partial=True)
        train_ds, other_ds = get_dataset(self.args) 
        self.train_ds = train_ds
        self.other_ds = other_ds

    def train_dataloader(self):
        """Returns the training DataLoader."""
        return DataLoader(
            self.train_ds, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.train_dataloader)

    def val_dataloader(self):
        """Returns the validation DataLoader."""
        return DataLoader(
            self.other_ds, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.val_dataloader)
    
    def test_dataloader(self):
        """Returns the test DataLoader."""
        return DataLoader(
            self.other_ds,
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.test_dataloader)
    
    def predict_dataloader(self):
        """Returns the predict DataLoader."""
        return DataLoader(
            self.other_ds, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.predict_dataloader)

class TrainingModule(pl.LightningModule):
    """Base TrainingModule which uses model, loss and metric setup functions defined in the 
    config file. The model is very close to a vanilla LightningModule.
    """
    def __init__(self, args: Config):
        """Args:
        args: Configuration dictionary.
        """
        super().__init__()
        self.args = args

        get_model = instantiate(self.args.model.setup, partial=True)
        self.model = get_model(self.args)

        loss_fn = LossCollection(args.losses)   
        self.train_loss = loss_fn.clone(prefix="train/")
        self.val_loss = loss_fn.clone(prefix="val/")
        
        # Performance Metrics
        get_metrics = instantiate(self.args.metrics, partial=True)
        metrics = get_metrics(self.args)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

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

        output = self.train_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

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

        output = self.val_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        return loss

    def test_step(self, batch):
        """Performs a test step."""
        x, y = batch
        logits = self(x)

        bs = x.size(0)
        output = self.test_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
    
    def predict_step(self, batch):
        """Performs a predict step."""
        return self(batch)

    def configure_optimizers(self):
        """Configures optimizers and learning rate schedulers based on the config file."""

        args_t = self.args.training
        optimizer_factory = instantiate(args_t.optimizer, partial=True)
        optimizer = optimizer_factory(self.parameters())
        
        output = {"optimizer": optimizer}

        if "lr_scheduler" in args_t:
            output["lr_scheduler"] = self.get_scheduler(optimizer)
        
        return output
    
    def get_scheduler(self, optimizer):
        """This function just needs to return a lr_scheduler_config dictionary as described in
        the Lightning docs. The base function here implements a complicated logic to handle
        almost all Pytorch schedulers by just changing the yaml configuration.
        """

        args_t = self.args.training
        scheduler_factory = instantiate(args_t.lr_scheduler.instance, partial=True)
        args = {"optimizer": optimizer}

        # Some lr_schedulers need to know the total number of iterations
        if getattr(args_t.lr_scheduler, "needs_total_iters", False):
            interval = getattr(args_t.lr_scheduler.scheduler_config, "interval", "step")
            if interval == "epoch":
                total_iters = self.trainer.max_epochs
            else:
                total_iters = self.trainer.estimated_stepping_batches
            if "OneCycleLR" in args_t.lr_scheduler.instance:
                # In OneCycleLR the parameter is named total_steps instead of total_iters
                args["total_steps"] = total_iters
            else:
                args["total_iters"] = total_iters 

        scheduler = scheduler_factory(**args)

        lr_scheduler_config = args_t.lr_scheduler.scheduler_config.to_dict()
        lr_scheduler_config["scheduler"] = scheduler

        return lr_scheduler_config

class Experiment:
    """Main class to run experiments based on a configuration file."""
    def __init__(self, args: Config):
        """Args:
        args: Configuration object.
        """
        self.args = args
        self.run_path = None
        self.model = None
        self.trainer = None

        self._set_seed()

        self.data_module = self.get_data_module()

    def setup_logging(self, stage: str = "fit") -> Path:
        """Handles the run_path name expansion and experiment directory creation logic as well 
        as lightning and wandb verbosity.
        """
        args_l = self.args.logging  

        if args_l.silence_lightning:
            silence_lightning()
        if _has_wandb and args_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        # Expand run_path name if {key} patterns are found
        args_copy = self.args.copy()
        args_copy.logging.run_path = ""
        run_path = generate_name_from_config(args_copy, args_l.run_path)
        run_path = Path(run_path)
        
        if stage == "fit" and args_l.create_folder:
            # Directory creation logic
            if run_path.exists():
                if args_l.overwrite_existing:
                    shutil.rmtree(run_path)  
                else:   
                    run_path = get_next_experiment_name(run_path)
            run_path.mkdir(parents=True, exist_ok=True)
            
            self.args.save(run_path / "config.yaml")

        return run_path

    def get_data_module(self) -> DataModule:
        """Override this for a different dataset logic (e.g.: multiple datasets)."""
        return DataModule(self.args)

    def get_model(self) -> TrainingModule:
        """Override this for a different model logic (e.g.: multiple models)."""
        return TrainingModule(self.args)

    def get_loggers(self):
        """Setup loggers."""

        args_l = self.args.logging
        run_path = self.run_path

        if run_path is None:
            raise ValueError("run_path must be set before setting loggers.")

        loggers: list[pl.loggers.LightningLoggerBase] = []

        if args_l.log_csv:
            loggers.append(
                CSVLogger(save_dir=run_path, name="", version="") 
            )
        
        args_w = args_l.wandb
        if args_w.log_wandb:
            if _has_wandb is False:
                raise ImportError("WandbLogger requires wandb to be installed. "
                                  "Please install it or disable wandb logging.")
            delete_wandb_run(args_w.wandb_project, str(run_path))
            loggers.append(WandbLogger(
                name=str(run_path), 
                save_dir=str(run_path),
                project=args_w.wandb_project, 
                group=args_w.wandb_group,
                config=self.args.to_dict()
            ))

        return loggers

    def get_callbacks(self, extra_callbacks: list[Callback] | None = None):
        """Constructs the list of callbacks. 
        
        extra_callbacks: passed from setup_trainer() (e.g. Optuna Pruning)
        """

        if self.run_path is None:
            raise ValueError("run_path must be set before setting callbacks.")

        args_l = self.args.logging
        args_t = self.args.training
        callbacks = extra_callbacks or []

        # Device Stats Monitor
        if args_t.monitor_device_stats:
            callbacks.append(DeviceStatsMonitor(cpu_stats=True))
        
        # Training Time Logger
        if args_l.log_training_time:
            callbacks.append(TrainingTimeLoggerCallback(self.run_path))
        
        # Custom Visualization & Plotting
        if args_l.save_val_data:
            callbacks.append(ImageSaveCallback(
                self.run_path, 
                args_l.val_data_indices, 
                log_disk=args_l.log_to_disk,
                mean=getattr(args_l, "dataset_mean", None),
                std=getattr(args_l, "dataset_std", None),
                log_wandb=args_l.wandb.log_wandb and args_l.wandb.log_val_data_wandb,
                class_labels=args_l.wandb.class_labels
                ))

        if args_l.log_plot:            
            callbacks.append(PlottingCallback())

        mode = "max" if args_t.maximize_validation_metric else "min"

        # Checkpointing
        if args_l.log_checkpoints:
            # Best Model
            if args_l.save_top_k_models > 0:
                callbacks.append(ModelCheckpoint(
                    dirpath=self.run_path / "models",
                    filename="best_model_epoch={epoch}_val_loss={val/loss:.4f}",
                    save_top_k=args_l.save_top_k_models,
                    save_last=False,
                    monitor=args_t.validation_metric,
                    mode=mode,
                    auto_insert_metric_name=False
                ))
            # Save model every n epochs
            if args_l.save_model_every_n_epochs > 0:
                callbacks.append(ModelCheckpoint(
                    dirpath=self.run_path / "models",
                    filename="last_checkpoint",
                    save_top_k=1,
                    save_last=False,
                    monitor=None,
                    every_n_epochs=args_l.save_model_every_n_epochs
                ))

        # Early Stopping
        if args_t.patience:
            callbacks.append(EarlyStopping(
                monitor=args_t.validation_metric, 
                patience=args_t.patience, 
                mode=mode,
                divergence_threshold=args_t.divergence_threshold
            ))

        # Instantiate extra callbacks from config
        if "extra_callbacks" in args_t:
            for cb_conf in args_t.extra_callbacks.values():
                cb = instantiate(cb_conf)
                callbacks.append(cb)

        return callbacks

    def get_profiler(self):
        """Setup profiler based on config."""

        if not getattr(self.args.training, "profile", False):
            return None

        verbosity = getattr(self.args.training, "profile_verbosity", 0)

        if verbosity >= 5:
            # TODO: check if this works
            experimental_config = torch.profiler._ExperimentalConfig( # type: ignore
                verbose=True,
                profiler_metrics=[
                    "kineto__cuda_core_occupancy", # GPU occupancy
                    "kineto__dram_throughput",     # GPU memory bandwidth
                ] 
            )
        else:
            experimental_config = None

        profiler = PyTorchProfiler(
        dirpath=str(self.run_path),
        filename="profile",
        export_to_chrome=True,
        with_flops = verbosity >= 1,
        record_shapes = verbosity >= 2,
        profile_memory = verbosity >= 3,
        with_stack = verbosity >= 4,
        experimental_config = experimental_config,
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=4, repeat=1)
        )

        return profiler

    def setup_trainer(self, extra_callbacks: list[Callback] | None = None) -> pl.Trainer:
        """Sets up the PyTorch Lightning Trainer."""


        loggers = self.get_loggers()
        callbacks = self.get_callbacks(extra_callbacks)   
        profiler = self.get_profiler()

        enable_checkpointing = any(isinstance(cb, ModelCheckpoint) for cb in callbacks) 

        args_t = self.args.training.trainer_params
        
        trainer = pl.Trainer(
            default_root_dir = str(self.run_path),
            devices = args_t.devices,
            strategy = args_t.strategy,
            precision = args_t.precision,
            max_epochs = args_t.max_epochs,
            accumulate_grad_batches = args_t.accumulate_grad_batches,
            gradient_clip_val = args_t.gradient_clip_val,
            check_val_every_n_epoch = args_t.check_val_every_n_epoch,
            log_every_n_steps = self.args.logging.log_every_n_steps,
            enable_progress_bar = self.args.logging.enable_progress_bar, 
            deterministic = args_t.deterministic,
            benchmark = args_t.benchmark,
            enable_checkpointing = enable_checkpointing,
            logger = loggers,
            callbacks = callbacks,
            profiler = profiler,
            enable_model_summary = False,
            **args_t.kwargs
        )

        return trainer

    def fit(
            self, 
            args_overrides: Config | dict | None = None, 
            extra_callbacks: list[Callback] | None = None,
            recreate_dataset: bool = False, 
            ):
        """Main entry point.

        Args:
            args_overrides: Dictionary to override args in self.args.
            recreate_dataset: If True, recreates the data module. Useful for hyperparameter
            optimization where dataset parameters may change.
            extra_loggers: Extra loggers to add to the trainer.
            extra_callbacks: Extra callbacks to add to the trainer.

        Returns:
            float: The value of the validation metric after training (useful for hyperparameter 
            optimization).
        """

        if args_overrides is not None:
            self.args.update_from(args_overrides)

        self._set_seed()

        if recreate_dataset:
            self.data_module = self.get_data_module()

        self.run_path = self.setup_logging(stage="fit")
        self.model = self.get_model()
        self.trainer = self.setup_trainer(extra_callbacks=extra_callbacks)

        self.trainer.fit(self.model, datamodule=self.data_module)
        
        # Return metric (Useful for Optuna)
        return self.trainer.callback_metrics[self.args.training.validation_metric].item()
    
    def test(
            self, 
            args_overrides: Config | dict | None = None, 
            checkpoint: str | None = None,
            recreate_dataset: bool = False, 
            ):
        """Test a model.

        Args:
            args_overrides: Dictionary to override args in self.args.
            checkpoint: Name of the checkpoint to load from the self.run_path / "models" folder. 
            It can also be "best" and "last", in which case ModelCheckpoint callbabks will be
            used to load the corresponding model.
            recreate_dataset: If True, recreates the data module. 
        """

        if args_overrides is not None:
            self.args.update_from(args_overrides)

        self._set_seed()

        if recreate_dataset:
            self.data_module = self.get_data_module()

        fit_called = self.run_path is not None
        if not fit_called:
            self.run_path = self.setup_logging(stage="test")
            self.model = self.get_model()
        # Recreate Trainer to allow different parameters between fit and test.
        self.trainer = self.setup_trainer()        

        if checkpoint is not None:
            checkpoint = self.run_path / "models" / checkpoint # type: ignore
        
        return self.trainer.test(
            self.model, datamodule=self.data_module, ckpt_path=checkpoint)
        
    def _set_seed(self):

        pl.seed_everything(self.args.seed, workers=True, verbose=False) 