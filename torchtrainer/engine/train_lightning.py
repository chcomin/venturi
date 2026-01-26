import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import Callback, DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import DataLoader

from torchtrainer.engine.composite import CompositeLoss
from torchtrainer.engine.config import Config, instantiate
from torchtrainer.engine.lightning_util import silence_lightning
from torchtrainer.engine.train_util import (
    delete_wandb_run,
    generate_name_from_config,
    get_next_experiment_name,
)

try:
    import wandb
except ImportError:
    _has_wandb = False
else:
    from pytorch_lightning.loggers import WandbLogger
    _has_wandb = True

torch.set_float32_matmul_precision("high")

class BaseDataModule(pl.LightningDataModule):
    """Base DataModule which uses a dataset setup function defined in the config file."""
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        # Dataset dictionary to hold train, val, test or predict datasets
        self.ds_dict: dict[str, torch.utils.data.Dataset] = {}

    def setup(self, stage=None):

        args_l = self.args.logging
        if args_l.silence_lightning:
            silence_lightning()
        if _has_wandb and args_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        # Call the function indicated in self.args.dataset.setup, passing the stage and args
        # stage can be 'fit', 'validate', 'test', or 'predict'
        get_dataset = instantiate(self.args.dataset.setup, partial=True)
        self.ds_dict = get_dataset(stage, self.args)

    def train_dataloader(self):
        if "train_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["train_ds"], 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.train_dataloader)

    def val_dataloader(self):
        if "val_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["val_ds"], 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.val_dataloader)
    
    def test_dataloader(self):
        if "test_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["test_ds"],
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.test_dataloader)
    
    def predict_dataloader(self):
        if "predict_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["predict_ds"], 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.predict_dataloader)

class BaseTrainingModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        get_model = instantiate(self.args.model.setup, partial=True)
        self.model = get_model(self.args)

        loss_fn = CompositeLoss(args.losses)   
        self.train_loss = loss_fn.clone(prefix="train/")
        self.val_loss = loss_fn.clone(prefix="val/")

        # Performance Metrics
        get_metrics = instantiate(self.args.metrics, partial=True)
        metrics = get_metrics(self.args)
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
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
        return self.validation_step(batch)
    
    def predict_step(self, batch):
        return self(batch)

    def configure_optimizers(self):

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

class PlottingCallback(Callback):
    """Reads the CSV log generated by Lightning and creates a static plot 
    (plots.png) at the end of every training epoch.
    """
    def on_train_epoch_end(self, trainer, pl_module):

        if trainer.global_rank != 0:
            return
               
        logger_dir = Path(trainer.log_dir)
        metrics_file = logger_dir / "metrics.csv"
        
        if not metrics_file.exists():
            return

        try:
            df = pd.read_csv(metrics_file)
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Loss Plot
            args_p = pl_module.args.logging.plot.left_plot
            has_line = False
            for key in args_p.metrics:
                key_p = key
                if key_p == "train/loss":
                    # Lightning automatically adds 'step' or 'epoch' suffixes to train loss
                    key_p = "train/loss_epoch"
                if key_p in df.columns:
                    data = df[[ "epoch", key_p]].dropna()
                    if len(data) > 1:
                        ax[0].plot(data["epoch"], data[key_p], label=key)
                        has_line = True
            
            ax[0].set_xlabel("epoch")
            if has_line:
                ax[0].legend()
            ax[0].set_ylim(args_p.ylim.min, args_p.ylim.max)
            
            # Metrics Plot
            args_p = pl_module.args.logging.plot.right_plot
            has_line = False
            for key in args_p.metrics:
                if key in df.columns:
                    data = df[[ "epoch", key]].dropna()
                    if len(data) > 1:
                        ax[1].plot(data["epoch"], data[key], label=key)
                        has_line = True

            ax[1].set_xlabel("epoch")
            if has_line:
                ax[1].legend()
            ax[1].set_ylim(args_p.ylim.min, args_p.ylim.max)
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
            run_path, 
            val_img_indices, 
            log_disk=True, 
            mean=None, 
            std=None,
            log_wandb=False, 
            class_labels=None
            ):
        super().__init__()

        run_path = Path(run_path)

        if log_disk:
            for idx in val_img_indices:
                save_dir = run_path / "images" / f"image_{idx}"
                save_dir.mkdir(parents=True, exist_ok=True)

        self.run_path = run_path
        self.val_img_indices = set(val_img_indices)
        self.log_disk = log_disk
        self.log_wandb = log_wandb
        self.mean = mean
        self.std = std
        self.class_labels = class_labels

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):

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
            logits = pl_module(inputs.to(pl_module.device))
            
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
                std = torch.tensor(self.std).view(-1, 1, 1)
                mean = torch.tensor(self.mean).view(-1, 1, 1)
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
                target = (target / target.max() * 255)
            if pred.max() > 0:
                pred = pred.float()
                pred = (pred / pred.max() * 255)
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
        
        class_labels = dict(enumerate(self.class_labels))

        import numpy as np
        unique_vals = np.unique(pred.byte().numpy())
        print(f"Mask Values: {unique_vals}") 
        print(f"Dict Keys: {class_labels.keys()}")

        # Create the WandB Image object
        wandb_image = wandb.Image(
            img.byte().numpy(), 
            masks={
                "predictions": {
                    "mask_data": pred.byte().numpy(),
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": target.byte().numpy(),
                    "class_labels": class_labels
                }
            },
            caption=f"Image {idx}"
        )

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log(
                    {f"val_predictions/image_{idx}": [wandb_image]}, 
                    step = trainer.global_step
                )          

class BaseExperiment:
    """Main class to run experiments based on a configuration file."""
    def __init__(self, args: Config):
        self.args = args
        self.run_path = None
        self.model = None
        self.data_module = None
        self.trainer = None

        args_l = self.args.logging
        if args_l.silence_lightning:
            silence_lightning()
        if _has_wandb and args_l.wandb.silence_wandb:
            os.environ["WANDB_SILENT"] = "True"

        self.data_module = self.get_data_module()
    
    def setup_logging(self) -> None:
        """Handles the run_path name expansion and experiment directory creation logic."""
        args_l = self.args.logging

        args_copy = self.args.copy()
        args_copy.logging.run_path = ""
        run_path = generate_name_from_config(args_copy, args_l.run_path)
        run_path = Path(run_path)

        if not args_l.create_folder:
            self.run_path = run_path
            return
        
        if run_path.exists():
            if args_l.overwrite_existing:
                shutil.rmtree(run_path)  
            else:   
                run_path = get_next_experiment_name(run_path)
        run_path.mkdir(parents=True, exist_ok=True)
            
        self.run_path = run_path
        
        self.args.save(run_path / "config.yaml")

    def get_data_module(self) -> BaseDataModule:
        """Override this for a different dataset logic (e.g.: multiple datasets)."""
        return BaseDataModule(self.args)

    def get_model(self) -> BaseTrainingModule:
        return BaseTrainingModule(self.args)

    def get_loggers(self, extra_loggers):
        """Setup loggers. Can be overridden to add MLFlow, Comet, etc."""
        # Use parent dir so we don't get nested version folders
        args_l = self.args.logging
        run_path = self.run_path
        
        loggers = []

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
            
        if extra_loggers:
            loggers.extend(extra_loggers)

        return loggers

    def get_callbacks(self, extra_callbacks=None):
        """Constructs the list of callbacks. 
        
        extra_callbacks: List[Callback] passed from run() (e.g. Optuna Pruning)
        """
        args_l = self.args.logging
        args_t = self.args.training
        callbacks = extra_callbacks or []
        
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
        # Early Stopping
        if args_t.patience:
            callbacks.append(EarlyStopping(
                monitor=args_t.validation_metric, 
                patience=args_t.patience, 
                mode=mode,
                divergence_threshold=args_t.divergence_threshold
            ))

        # Checkpointing
        if args_l.log_checkpoints:
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

        # Device Stats Monitor
        if args_t.monitor_device_stats:
            callbacks.append(DeviceStatsMonitor(cpu_stats=True))

        if "extra_callbacks" in args_t:
            for cb_conf in args_t.extra_callbacks.values():
                cb = instantiate(cb_conf)
                callbacks.append(cb)

        return callbacks

    def get_profiler(self):

        verbosity = getattr(self.args.training, "profile_verbosity", 0)

        if verbosity >= 5:
            experimental_config = torch.profiler._ExperimentalConfig(
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

    def run(
            self, 
            args_overrides = None, 
            recreate_dataset=False, 
            extra_loggers=None, 
            extra_callbacks=None
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

        if args_overrides:
            self.args.update_from_dict(args_overrides)

        self.setup_logging()

        pl.seed_everything(self.args.seed, workers=True, verbose=False)

        if recreate_dataset:
            self.data_module = self.get_data_module()
        
        self.model = self.get_model()
        
        # Configure Trainer components
        loggers = self.get_loggers(extra_loggers)
        callbacks = self.get_callbacks(extra_callbacks)   

        profiler = None
        if getattr(self.args.training, "profile", False):
            profiler = self.get_profiler()

        enable_checkpointing = any(isinstance(cb, ModelCheckpoint) for cb in callbacks) 

        args_t = self.args.training.trainer_params
        
        self.trainer = pl.Trainer(
            default_root_dir = str(self.run_path),
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

        # Fit
        self.trainer.fit(self.model, datamodule=self.data_module)
        
        # Return metric (Useful for Optuna)
        return self.trainer.callback_metrics.get(
            self.args.training.validation_metric, torch.tensor(0.0)).item()

