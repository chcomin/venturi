import argparse
import ast
import logging
import random
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torchvision
import yaml
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
except ImportError:
    _has_wandb = False
else:
    from pytorch_lightning.loggers import WandbLogger
    _has_wandb = True

class LightningLogFilter(logging.Filter):
    """Filtra mensagens específicas e teimosas do PyTorch Lightning."""
    
    def filter(self, record):
        msg = str(record.msg)

        forbidden_phrases = [
            "Seed set to",
            "LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES",
        ]
        
        # Verify if any forbidden phrase is in the message
        return not any(phrase in msg for phrase in forbidden_phrases)

def silence_lightning():

    warnings.filterwarnings("ignore", ".*exists and is not empty.*")
    warnings.filterwarnings("ignore", ".*The number of training batches.*")

    # Instancia o filtro
    lightning_filter = LightningLogFilter()
    
    # Aplica o filtro na raiz do Lightning (para casos bem comportados)
    logging.getLogger("lightning.pytorch").addFilter(lightning_filter)
    logging.getLogger("lightning.fabric").addFilter(lightning_filter)
    
    # A "Vacinação em Massa": Itera sobre todos os loggers ativos
    # Isso pega os loggers rebeldes como 'lightning.fabric.utilities.seed'
    for logger_name in logging.Logger.manager.loggerDict:
        if "lightning" in logger_name:
            logger = logging.getLogger(logger_name)
            logger.addFilter(lightning_filter)
            logger.setLevel(logging.ERROR)

def seed_worker(worker_id):
    """Ensures dataloader workers are seeded correctly."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ==========================================
# 1. Custom Argparse Actions
# ==========================================
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            # Attempt to convert to basic types
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass # Keep as string
            getattr(namespace, self.dest)[key] = value

class ParseText(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, " ".join(values))

# ==========================================
# 2. Argument Parser (From your snippet)
# ==========================================
def get_parser() -> argparse.ArgumentParser:
    """Get the argument parser for the script."""
    config_parser = argparse.ArgumentParser(description="Training Config", add_help=False)
    config_parser.add_argument("--config", default="", metavar="FILE", 
        help="Path to YAML config file specifying default arguments")

    parser = argparse.ArgumentParser(
        description="Below, N represents integer values and V represents float values")

    # Logging parameters
    group = parser.add_argument_group("Logging parameters")
    group.add_argument("-p", "--experiments_path", default="experiments", metavar="PATH", 
                       help="Path to save experiments data")
    group.add_argument("-e", "--experiment_name", default="default_exp", 
                       metavar="NAME", help="Name of the experiment")
    group.add_argument("-n", "--run_name", default="default_run", metavar="NAME", 
                       help="Name of the run for a given experiment")
    group.add_argument("--validate_every", type=int, default=1, metavar="N", 
                       help="Run a validation step every N epochs")
    group.add_argument("--save_val_imgs", action="store_true", 
                       help="Save some validation images when validating")
    group.add_argument("--val_img_indices", nargs="*", type=int, default=(0,), 
                       metavar="N N N", help="Indices of the validation images to save")
    group.add_argument("--suppress_checkpoint", action="store_true",
               help="Suppress model checkpoint saving at the end of each epoch.")
    group.add_argument("--copy_model_every", type=int, default=0, metavar="N", 
                       help="Save a copy of the model every N epochs.")
    group.add_argument("--suppress_best_checkpoint", action="store_true", 
                       help="Avoid saving the best checkpoint of the model.")
    group.add_argument("--log_wandb", action="store_true", 
                       help="If wandb should also be used for logging.")
    group.add_argument("--wandb_project", default="uncategorized", 
                       help="Name of the wandb project to log the data.")
    group.add_argument("--wandb_group", default="", nargs="*", action=ParseText, 
                       help="Name of the wandb group to log the data.")
    parser.add_argument("--log_images_wandb", action="store_true", 
                        help="If wandb should be used for logging validation images.")
    group.add_argument("--disable_tqdm", action="store_true", 
                       help="Disable tqdm progressbar.")
    parser.add_argument("--meta", default="", nargs="*", action=ParseText, 
                        help="Additional metadata.")

    # Dataset parameters
    group = parser.add_argument_group("Dataset parameters")
    group.add_argument("--dataset_path", default="./data", help="Path to the dataset root directory")        
    group.add_argument("--dataset_class", default="MyDataset", help="Name of the dataset class to use")
    group.add_argument("--split_strategy", default="0.2",  metavar="STRING",
                       help="How to split the data into train/val.")
    group.add_argument("--augmentation_strategy", default=None, metavar="STRING", 
                       help="Data augmentation procedure.")
    group.add_argument("--resize_size", default=(384,384), nargs=2, type=int, 
                       metavar=("N", "N"), help="Size to resize the images.")
    group.add_argument("--dataset_params", nargs="*", default={}, action=ParseKwargs, 
        metavar="par1=v1", help="Additional parameters for dataset.")
    group.add_argument("--loss_function", default="cross_entropy", metavar="LOSS", 
                       help="Loss function to use during training")
    group.add_argument("--ignore_class_weights", action="store_true", 
                       help="If provided, ignore class weights for the loss function")

    # Model parameters
    group = parser.add_argument_group("Model parameters")
    group.add_argument("--model_class", default="MyModel", help="Name of the model to train")
    group.add_argument("--weights_strategy", default=None, metavar="STRING", 
                       help="Method to load weights")
    group.add_argument("--model_params", nargs="*", default={}, action=ParseKwargs, 
                       metavar="par1=v1", help="Additional parameters for model.")

    # Training parameters
    group = parser.add_argument_group("Training parameters")
    group.add_argument("--num_epochs", type=int, default=2, metavar="N", 
                       help="Number of training epochs")
    group.add_argument("--validation_metric", default="val_loss", nargs="*", 
                       metavar="METRIC", action=ParseText, 
                       help="Which metric to use for early stopping")
    group.add_argument("--patience", type=int, default=None, metavar="N", 
                       help="Early stopping patience.")
    group.add_argument("--maximize_validation_metric", action="store_true", 
                       help="If set, early stopping will maximize.")
    group.add_argument("--lr", type=float, default=0.01, metavar="V", 
                       help="Initial learning rate")
    group.add_argument("--lr_decay", type=float, default=1., metavar="V", 
                       help="Learning rate decay")
    group.add_argument("--bs_train", type=int, default=32, metavar="N", 
                       help="Batch size used durig training")
    group.add_argument("--bs_valid", type=int, default=8, metavar="N", 
                       help="Batch size used durig validation")
    group.add_argument("--weight_decay", type=float, default=1e-4, metavar="V", 
                       help="Weight decay for the optimizer")
    group.add_argument("--optimizer", default="sgd", help="Optimizer to use")
    group.add_argument("--momentum", type=float, default=0.9, metavar="V", 
                       help="Momentum/beta1 of the optimizer")
    group.add_argument("--seed", type=int, default=0, metavar="N", 
                       help="Seed for the random number generator")

    # Device and efficiency parameters
    group = parser.add_argument_group("Device and efficiency parameters")
    group.add_argument("--num_workers", type=int, default=5, metavar="N", 
                       help="Number of workers for the DataLoader")
    group.add_argument("--pin_memory", action="store_false", 
                       help="If DataLoader should pin memory.")
    group.add_argument("--use_amp", action="store_true", 
                       help="If automatic mixed precision should be used")
    group.add_argument("--deterministic", action="store_true", 
                       help="If deterministic algorithms should be used")
    group.add_argument("--benchmark", action="store_true", 
                       help="If cuda benchmark should be used")
    group.add_argument("--profile", action="store_true", 
                       help="If set, enable the profile mode.")
    group.add_argument("--profile_batches", type=int, default=3, metavar="N", 
                       help="Number of batches to profile.")
    group.add_argument("--profile_verbosity", type=int, default=0, metavar="N", 
                       help="Profile verbosity.")

    return parser, config_parser

# ==========================================
# 3. Utilities (Factories)
# ==========================================
def get_loss(name, ignore_weights=False):
    # This is a placeholder. You would typically return specific torch.nn modules here.
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")

def get_optimizer(model_params, args):
    args_t = args.training
    if args_t.optimizer.lower() == "sgd":
        return SGD(
            model_params, lr=args_t.lr, momentum=args_t.momentum, weight_decay=args_t.weight_decay)
    elif args_t.optimizer.lower() == "adam":
        return Adam(model_params, lr=args_t.lr, weight_decay=args_t.weight_decay)
    elif args_t.optimizer.lower() == "adamw":
        return AdamW(model_params, lr=args_t.lr, weight_decay=args_t.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args_t.optimizer}")

# ==========================================
# 4. Lightning System
# ==========================================
class TrainingModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        args = Config(args)
        self.args = args
        args_d  = args.dataset
        num_classes = args_d.num_classes

        # Placeholder for demo purposes:
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
        # 2. Loss
        self.loss_fn = get_loss(args_d.loss_function, args_d.ignore_class_weights)

        metrics = torchmetrics.MetricCollection({
            "Accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, ignore_index=-100),
            "IoU": torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-100),
            #"Dice": torchmetrics.segmentation.DiceScore(num_classes=num_classes),
            "Precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, ignore_index=-100),
            "Recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, ignore_index=-100),
        })

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        output = self.train_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        output = self.val_metrics(logits, y)
        self.log_dict(output, on_step=False, on_epoch=True, prog_bar=False)
        
        # Handle Image Saving logic if enabled
        if self.args.logging.save_val_imgs and batch_idx == 0:
            pass
            #self._log_images(x, y, logits)
            
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), self.args)
        
        if self.args.training.lr_decay > 0:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, 
                total_iters=self.trainer.estimated_stepping_batches, 
                power=self.args.training.lr_decay
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
            }
            return [optimizer], [lr_scheduler_config]
        
        return optimizer
    
# ==========================================
# 5. Data Module
# ==========================================
class GenericDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        # Demo Placeholder Dataset
        args_d = self.args.dataset
        H, W = args_d.resize_size
        N_SAMPLES = 50 # Small number for mockup
        
        # MOCK INPUT: Random RGB Images (N, 3, H, W)
        mock_images = torch.rand(N_SAMPLES, 3, H, W)
        
        # MOCK TARGET: Random Segmentation Masks (N, H, W) with values 0 to num_classes-1
        mock_targets = torch.randint(0, args_d.num_classes, (N_SAMPLES, H, W))
        
        full_dataset = TensorDataset(mock_images, mock_targets)

        # Split strategy
        try:
            val_split = float(args_d.split_strategy)
            val_size = int(len(full_dataset) * val_split)
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(self.args.seed)
            self.train_ds, self.val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
        except ValueError:
            print("Complex split strategy not implemented in this demo")
            self.train_ds = full_dataset
            self.val_ds = full_dataset

    def train_dataloader(self):
        # TODO: collate_fn
        return DataLoader(
            self.train_ds, 
            batch_size=self.args.training.bs_train, 
            shuffle=True, 
            num_workers=self.args.performance.num_workers,
            persistent_workers=self.args.performance.num_workers > 0,
            worker_init_fn=seed_worker,
            pin_memory=self.args.performance.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.args.training.bs_valid, 
            shuffle=False, 
            num_workers=self.args.performance.num_workers,
            persistent_workers=self.args.performance.num_workers > 0,
            worker_init_fn=seed_worker,
            pin_memory=self.args.performance.pin_memory,
        )

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
    
    def setup_dirs(self, interactive=True):
        """Handles the directory creation logic. 
        interactive=False is useful for automated jobs (Optuna/Slurm).
        """
        args_l = self.args.logging
        experiments_path = Path(args_l.experiments_path)
        run_name = args_l.run_name
        
        # If we are in an Optuna trial, we might want to auto-generate names or skip checks
        if not interactive:
            # Setup path without asking questions
            experiment_path = experiments_path / args_l.experiment_name
            run_path = experiment_path / run_name
            run_path.mkdir(parents=True, exist_ok=True)
            self.run_path = run_path
            return

        experiment_path = experiments_path / args_l.experiment_name
        run_path = experiment_path / run_name

        if run_path.exists():
            print(f"Run directory '{run_path}' already exists.")
            action = input(
                "Press Enter to overwrite, write 'exit' to cancel, or provide a new name: ").strip()
            
            if action == "exit":
                sys.exit(0)
            elif action:
                run_name = action
                run_path = experiment_path / run_name
            else:
                shutil.rmtree(run_path)

        run_path.mkdir(parents=True, exist_ok=True)
        
        # Create subfolders
        if args_l.save_val_imgs:
            (run_path / "images").mkdir(exist_ok=True)
            for idx in args_l.val_img_indices:
                (run_path / "images" / f"image_{idx}").mkdir(exist_ok=True)
                
        if args_l.copy_model_every and not args_l.suppress_checkpoint:
            (run_path / "models").mkdir(exist_ok=True)
            
        self.run_path = run_path
        
        # Save config
        with open(run_path / "config.yaml", "w") as f:
            yaml.dump(vars(self.args), f)

    def get_data_module(self):
        """Override this for different dataset logic"""
        return GenericDataModule(self.args)

    def get_model(self):
        """Override this for different model logic"""
        # Example of dynamic num_classes detection
        return TrainingModule(self.args.to_dict())

    def get_loggers(self):
        """Setup loggers. Can be overridden to add MLFlow, Comet, etc."""
        # Use parent dir so we don't get nested version folders
        args_l = self.args.logging
        save_dir = str(self.run_path.parent)
        
        loggers = []

        loggers.append(
            CSVLogger(save_dir=save_dir, name=args_l.run_name, version=None)
        )
        
        if args_l.log_wandb:
            if _has_wandb is False:
                raise ImportError("WandbLogger requires wandb to be installed. Please install it or disable wandb logging.")
            loggers.append(WandbLogger(
                project=args_l.wandb_project, 
                name=args_l.run_name, 
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
        
        # 1. Custom Visualization & Plotting
        if args_l.save_val_imgs:
            callbacks.append(ImageSaveCallback(
                self.run_path, args_l.val_img_indices, log_wandb=args_l.log_images_wandb))
        callbacks.append(PlottingCallback())

        # 2. Early Stopping
        if args_t.patience:
            mode = "max" if args_t.maximize_validation_metric else "min"
            callbacks.append(EarlyStopping(
                monitor=args_t.validation_metric, 
                patience=args_t.patience, 
                mode=mode
            ))

        # 3. Checkpointing
        if not args_l.suppress_checkpoint:
            mode = "max" if args_t.maximize_validation_metric else "min"
            # Best Model
            if not args_l.suppress_best_checkpoint:
                callbacks.append(ModelCheckpoint(
                    dirpath=self.run_path,
                    filename="best_model",
                    monitor=args_t.validation_metric,
                    mode=mode,
                    save_top_k=1
                ))
            # Last Model
            callbacks.append(ModelCheckpoint(
                dirpath=self.run_path,
                filename="last_checkpoint",
                save_last=True
            ))
            if args_l.copy_model_every > 0:
                callbacks.append(ModelCheckpoint(
                    dirpath=self.run_path / "models",
                    filename="checkpoint_{epoch}",
                    every_n_epochs=args_l.copy_model_every,
                    save_top_k=-1 # Keep all
                ))
            
        return callbacks

    def run(self, interactive=True, extra_callbacks=None, recreate_dataset=False):
        """Main entry point.
        """
        # 1. Setup Phase
        self.setup_dirs(interactive=interactive)
        pl.seed_everything(self.args.seed)

        if recreate_dataset:
            self.data_module = self.get_data_module()
        
        self.model = self.get_model()
        
        # 2. Configure Trainer components
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
        
        # 3. Initialize Trainer
        self.trainer = pl.Trainer(
            default_root_dir=str(self.run_path),
            logger=loggers,
            callbacks=callbacks,
            max_epochs=self.args.training.num_epochs,
            check_val_every_n_epoch=self.args.logging.validate_every,
            accelerator="auto",
            devices=1, 
            enable_progress_bar=not self.args.logging.disable_tqdm, # Disable tqdm for Optuna (non-interactive)
            enable_checkpointing=True,
            precision="16-mixed" if self.args.performance.use_amp else 32,
            deterministic=self.args.performance.deterministic,
            benchmark=self.args.performance.benchmark,
            profiler=profiler,
            enable_model_summary=interactive
        )

        # 4. Fit
        self.trainer.fit(self.model, datamodule=self.data_module)
        
        # 5. Return metric (Useful for Optuna)
        return self.trainer.callback_metrics.get(
            self.args.training.validation_metric, torch.tensor(0.0)).item()

if __name__ == "__main__":

    from config.config_util import Config

    args = Config("config/default.yaml")

    # 2. Run Experiment
    runner = ExperimentRunner(args)
    final_metric = runner.run(interactive=False)
