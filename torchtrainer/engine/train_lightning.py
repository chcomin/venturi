import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.segmentation
import torchvision
import yaml
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, default_collate

from torchtrainer.engine.config import Config, instantiate
from torchtrainer.engine.lightning_util import silence_lightning
from torchtrainer.engine.train_util import seed_worker, generate_name_from_config, get_next_experiment_name
from torchtrainer.engine.composite import CompositeLoss

try:
    import wandb
except ImportError:
    _has_wandb = False
else:
    from pytorch_lightning.loggers import WandbLogger
    _has_wandb = True

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ds_dict: dict[str, torch.utils.data.Dataset] = {}

    def setup(self, stage=None):

        # Call the function indicated in self.args.dataset.setup, passing self.args as
        # the only parameter
        get_dataset = instantiate(self.args.dataset.setup, partial=True)
        self.ds_dict = get_dataset(stage, self.args)

    def train_dataloader(self):
        if "train_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["train_ds"], 
            worker_init_fn=seed_worker, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.train_dataloader)

    def val_dataloader(self):
        if "val_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["val_ds"],
            worker_init_fn=seed_worker, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.val_dataloader)
    
    def test_dataloader(self):
        if "test_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["test_ds"],
            worker_init_fn=seed_worker, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.test_dataloader)
    
    def predict_dataloader(self):
        if "predict_ds" not in self.ds_dict:
            return None
        return DataLoader(
            self.ds_dict["predict_ds"],
            worker_init_fn=seed_worker, 
            generator=torch.Generator().manual_seed(self.args.seed),
            **self.args.dataset.predict_dataloader)

class BaseTrainingModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        get_model = instantiate(self.args.model.setup, partial=True)
        self.model = get_model(self.args)

        self.loss_fn = CompositeLoss(args.losses)   
        # Performance Metrics
        metrics = self.get_metrics()

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, loss_logs = self.loss_fn(logits, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for key, value in loss_logs.items():
            # Add train_ prefix to loss names
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=False)

        output = self.train_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss, loss_logs = self.loss_fn(logits, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for key, value in loss_logs.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=False)

        output = self.val_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):

        args_t = self.args.training
        optimizer_factory = instantiate(self.args.training.optimizer, partial=True)
        optimizer = optimizer_factory(self.parameters())
        
        output = {"optimizer": optimizer}

        if "lr_scheduler" in args_t:
            scheduler_factory = instantiate(args_t.lr_scheduler.instance, partial=True)
            args = {"optimizer": optimizer}
            # Some lr_schedulers need to know the total number of iterations
            if getattr(args_t.lr_scheduler, "needs_total_iters", False):
                if args_t.lr_scheduler.scheduler_config.interval == "epoch":
                    total_steps = self.trainer.max_epochs
                else:
                    total_steps = self.trainer.estimated_stepping_batches
                args["total_iters"] = total_steps

            scheduler = scheduler_factory(**args)

            lr_scheduler_config = args_t.lr_scheduler.scheduler_config.to_dict()
            lr_scheduler_config["scheduler"] = scheduler

            output["lr_scheduler"] = lr_scheduler_config
        
        return output

    def get_metrics(self):
        # Metrics are usually not hyperparameters, so no need to define them in the yaml file.
        num_classes = self.args.dataset.params.num_classes

        metrics = torchmetrics.MetricCollection({
            "Accuracy": torchmetrics.Accuracy(task="binary", num_classes=num_classes),
            "IoU": torchmetrics.JaccardIndex(task="binary", num_classes=num_classes),
            "Dice": torchmetrics.segmentation.DiceScore(num_classes=num_classes, average="macro"),
            "Precision": torchmetrics.Precision(task="binary", num_classes=num_classes),
            "Recall": torchmetrics.Recall(task="binary", num_classes=num_classes),
        })

        return metrics

class PlottingCallback(Callback):
    """Reads the CSV log generated by Lightning and creates a static plot 
    (plots.png) at the end of every validation epoch.
    """
    def on_validation_epoch_end(self, trainer, pl_module):

        if trainer.global_rank != 0:
            return
        
        logger_dir = Path(trainer.log_dir)
        metrics_file = logger_dir / "metrics.csv"
        
        if not metrics_file.exists():
            return

        try:
            df = pd.read_csv(metrics_file)
            
            # Setup Plotting (similar to your LoggerPlotter)
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # 1. Loss Plot
            # Group keys by what you want to plot together
            if "train_loss" in df.columns:
                # Interpolate to handle different step logging frequencies
                ax[0].plot(df["epoch"], df["train_loss"].interpolate(), label="Train Loss")
            if "val_loss" in df.columns:
                # Drop NAs because val_loss is usually logged less frequently (once per epoch)
                val_data = df[["epoch", "val_loss"]].dropna()
                ax[0].plot(val_data["epoch"], val_data["val_loss"], label="Val Loss")
            
            ax[0].set_title("Losses")
            ax[0].set_xlabel("Epoch")
            ax[0].legend()
            
            # 2. Metrics Plot
            metric_keys = ["val_Accuracy", "val_IoU", "val_Dice"]
            for key in metric_keys:
                if key in df.columns:
                    data = df[["epoch", key]].dropna()
                    ax[1].plot(data["epoch"], data[key], label=key)
            
            ax[1].set_title("Metrics")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylim(0, 1)
            ax[1].legend()

            # Save
            plt.tight_layout()
            plt.savefig(logger_dir / "plots.png")
            plt.close(fig)
        
        except Exception as e:
            print(f"Error generating plot: {e}")

class ImageSaveCallback(Callback):
    """Saves validation predictions to disk, matching the folder structure 
    of the old script: run_path/images/image_{idx}/epoch_{epoch}.png
    """
    def __init__(self, run_path, val_img_indices, log_wandb=False, mean=None, std=None):
        super().__init__()
        self.run_path = Path(run_path)
        self.val_img_indices = set(val_img_indices)
        self.log_wandb = log_wandb
        # Add normalization params if your dataset uses them, to denormalize before saving
        self.mean = mean
        self.std = std

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # 1. Unpack batch
        imgs, targets = batch
        
        # 2. Calculate global indices for this batch
        batch_size = imgs.size(0)
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size
        
        # 3. Check if any target index is in this batch
        # Range is [start_idx, end_idx)
        current_indices = set(range(start_idx, end_idx))
        intersect = self.val_img_indices.intersection(current_indices)
        
        if not intersect:
            return

        # 4. Get Predictions (Forward Pass)
        # We assume the model outputs raw logits
        with torch.no_grad():
            logits = pl_module(imgs.to(pl_module.device))
            
            # Apply argmax for multiclass or sigmoid>0.5 for binary
            # Adjust this based on your specific model output (Binary vs Multi)
            if logits.shape[1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
            else:
                preds = torch.argmax(logits, dim=1).unsqueeze(1).float()

        # 5. Loop through interesting indices and save
        for global_idx in intersect:
            local_idx = global_idx - start_idx
            
            # Get Tensors
            img_tensor = imgs[local_idx].cpu()
            target_tensor = targets[local_idx].cpu()
            pred_tensor = preds[local_idx].cpu()

            # --- Visualization Logic (Customizable) ---
            # This creates a side-by-side: Input | Target | Pred
            
            # Denormalize image if necessary (simple Example)
            if self.mean is not None and self.std is not None:
                # Denormalize: image = (image * std) + mean
                # Ensure shapes match (C, 1, 1) for broadcasting
                std = torch.tensor(self.std).view(-1, 1, 1)
                mean = torch.tensor(self.mean).view(-1, 1, 1)
                img_tensor = img_tensor * std + mean

            self._save_to_disk(trainer, global_idx, img_tensor, target_tensor, pred_tensor)
            if self.log_wandb and isinstance(trainer.logger, WandbLogger):
                self._log_to_wandb(trainer, global_idx, img_tensor, target_tensor, pred_tensor)


    def _save_to_disk(self, trainer, idx, img, target, pred):

            # Create specific directory: run_path/images/image_{idx}
            save_dir = self.run_path / "images" / f"image_{idx}"
            save_dir.mkdir(parents=True, exist_ok=True)

            # Convert to PIL
            to_pil = torchvision.transforms.ToPILImage()
            
            # Ensure proper shapes for visualization (C, H, W)
            # If target is (H, W), unsqueeze to (1, H, W)
            if target.ndim == 2:
                target = target.unsqueeze(0)
            
            # Helper to normalize masks to 0-255 for visibility
            def prep_mask(m):
                m = m.float()
                if m.max() > 1e-6:
                    m = m / m.max() 
                return m
            
            if target.ndim == 2: target = target.unsqueeze(0)

            img_pil = to_pil(img)
            target_pil = to_pil(prep_mask(target))
            pred_pil = to_pil(prep_mask(pred))
            # Stitch them together
            w, h = img_pil.size
            combined = Image.new("RGB", (w * 3, h))
            combined.paste(img_pil, (0, 0))
            combined.paste(target_pil.convert("RGB"), (w, 0))
            combined.paste(pred_pil.convert("RGB"), (w * 2, 0))

            # Save
            filename = f"epoch_{trainer.current_epoch}.png"
            combined.save(save_dir / filename)

    def _log_to_wandb(self, trainer, idx, img, target, pred):
        """Logs an interactive segmentation mask. 
        WandB allows you to toggle class overlays on/off in the UI.
        """
        
        # 1. Define class labels (optional, improves UI)
        class_labels = {0: "Background", 1: "Vessel", 2: "Artifact"} # Update for your case

        # 2. Create the WandB Image object
        wandb_image = wandb.Image(
            img, 
            masks={
                "predictions": {
                    "mask_data": pred.numpy(),
                    "class_labels": class_labels
                },
                "ground_truth": {
                    "mask_data": target.numpy(),
                    "class_labels": class_labels
                }
            },
            caption=f"Image {idx}"
        )

        # 3. Log it
        # We use 'global_step' to align with the training curves
        trainer.logger.experiment.log({
            f"val_predictions/image_{idx}": wandb_image, 
            "epoch": trainer.current_epoch
        })          

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.run_path = None
        self.model = None
        self.data_module = None
        self.trainer = None

        torch.set_float32_matmul_precision("high")
        #silence_lightning()

        self.data_module = self.get_data_module()
    
    def setup_dirs(self):
        """Handles the directory creation logic. 
        interactive=False is useful for automated jobs (Optuna/Slurm).
        """
        args_l = self.args.logging

        args_copy = self.args.copy()
        args_copy.logging.run_path = ""
        run_path = generate_name_from_config(args_copy, args_l.run_path)
        run_path = Path(run_path)
        
        if run_path.exists():
            if args_l.overwrite_existing:
                shutil.rmtree(run_path)  
            else:   
                run_path = get_next_experiment_name(run_path)
        run_path.mkdir(parents=True, exist_ok=True)

        # Create subfolders
        if args_l.save_val_imgs:
            (run_path / "images").mkdir(exist_ok=True)
            for idx in args_l.val_img_indices:
                (run_path / "images" / f"image_{idx}").mkdir(exist_ok=True)
                
        (run_path / "models").mkdir(exist_ok=True)
            
        self.run_path = run_path
        
        # Save config
        self.args.save(run_path / "config.yaml")

    def get_data_module(self):
        """Override this for a different dataset logic (e.g.: multiple datasets)."""
        return BaseDataModule(self.args)

    def get_model(self):
        return BaseTrainingModule(self.args)

    def get_loggers(self):
        """Setup loggers. Can be overridden to add MLFlow, Comet, etc."""
        # Use parent dir so we don't get nested version folders
        args_l = self.args.logging
        save_dir = str(self.run_path.parent)
        run_name = self.run_path.name
        
        loggers = []

        loggers.append(
            CSVLogger(save_dir=save_dir, name=run_name, version=None)
        )
        
        if args_l.log_wandb:
            if _has_wandb is False:
                raise ImportError("WandbLogger requires wandb to be installed. Please install it or disable wandb logging.")
            loggers.append(WandbLogger(
                project=args_l.wandb_project, 
                name=run_name, 
                save_dir=str(self.run_path)
            ))
            
        return loggers

    def get_callbacks(self, extra_callbacks=None):
        """Constructs the list of callbacks. 
        extra_callbacks: List[Callback] passed from run() (e.g. Optuna Pruning)
        """
        args_l = self.args.logging
        args_t = self.args.training
        callbacks = extra_callbacks or []
        
        # Custom Visualization & Plotting
        if args_l.save_val_imgs:
            callbacks.append(ImageSaveCallback(
                self.run_path, args_l.val_img_indices, log_wandb=args_l.log_images_wandb))
        callbacks.append(PlottingCallback())

        # Early Stopping
        if args_t.patience:
            mode = "max" if args_t.maximize_validation_metric else "min"
            callbacks.append(EarlyStopping(
                monitor=args_t.validation_metric, 
                patience=args_t.patience, 
                mode=mode
            ))

        # Checkpointing
        mode = "max" if args_t.maximize_validation_metric else "min"

        if args_l.save_every_n_epochs > 0:
            callbacks.append(ModelCheckpoint(
                dirpath=self.run_path / "models",
                filename="last_checkpoint",
                save_top_k=1,
                save_last=False,
                monitor=None,
                every_n_epochs=args_l.save_every_n_epochs
            ))
        # Best Model
        if args_l.save_top_k_models > 0:
            callbacks.append(ModelCheckpoint(
                dirpath=self.run_path / "models",
                filename="best_model_{epoch}_{val_loss:.4f}",
                save_top_k=args_l.save_top_k_models,
                save_last=False,
                monitor=args_t.validation_metric,
                mode=mode
            ))

        if "extra_callbacks" in args_t:
            for cb_conf in args_t.extra_callbacks.values():
                cb = instantiate(cb_conf)
                callbacks.append(cb)

        return callbacks

    def run(self, args_overrides = None, extra_callbacks=None, recreate_dataset=False):
        """Main entry point."""

        if args_overrides:
            self.args.update_from_dict(args_overrides)

        # Setup Phase
        if self.args.logging.enable:
            self.setup_dirs()
        pl.seed_everything(self.args.seed)

        if recreate_dataset:
            self.data_module = self.get_data_module()
        
        self.model = self.get_model()
        
        # Configure Trainer components
        loggers = self.get_loggers()
        callbacks = self.get_callbacks(extra_callbacks)   

        profiler = None
        if self.args.performance.profile:
            profiler = PyTorchProfiler(
            dirpath=str(self.run_path),
            filename="profile",
            export_to_chrome=True,
            with_stack=(self.args.performance.profile_verbosity > 1),
            with_shapes=(self.args.performance.profile_verbosity > 0),
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=self.args.performance.profile_batches)
            )

        enable_checkpointing = any(isinstance(cb, ModelCheckpoint) for cb in callbacks) 
        
        self.trainer = pl.Trainer(
            default_root_dir=str(self.run_path),
            logger=loggers,
            callbacks=callbacks,
            max_epochs=self.args.training.num_epochs,
            check_val_every_n_epoch=self.args.logging.validate_every,
            accelerator="auto",
            devices=self.args.performance.devices, 
            enable_progress_bar=not self.args.logging.disable_tqdm, 
            enable_checkpointing=enable_checkpointing,
            precision="16-mixed" if self.args.performance.use_amp else 32,
            deterministic=self.args.performance.deterministic,
            benchmark=self.args.performance.benchmark,
            profiler=profiler,
            enable_model_summary=False
        )

        # Fit
        self.trainer.fit(self.model, datamodule=self.data_module)
        
        # Return metric (Useful for Optuna)
        return self.trainer.callback_metrics.get(
            self.args.training.validation_metric, torch.tensor(0.0)).item()

