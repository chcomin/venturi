import torch
import torchmetrics
from torchmetrics.classification import BinaryF1Score

from torchtrainer.engine.config import Config


class Dice(BinaryF1Score):
    """Dice score for binary segmentation."""
    def __init__(self, threshold=0.5, ignore_index=None, **kwargs):
        # Force 'samplewise' to calculate Dice per image
        super().__init__(
            threshold=threshold, 
            multidim_average="samplewise", 
            ignore_index=ignore_index,
            **kwargs
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if target.is_floating_point():
            target = target.long()
        super().update(preds, target)

    def compute(self) -> torch.Tensor:
        val = super().compute()
        return val.mean()

def binary_segmentation_metrics(args: Config) -> torchmetrics.MetricCollection:
    """ Define common metrics for binary segmentation tasks.

    Args:
        args (Config): Configuration object containing hyperparameters.
    """

    metrics = torchmetrics.MetricCollection({
        "accuracy": torchmetrics.Accuracy(task="binary"),
        "dice": Dice(),
        "precision": torchmetrics.Precision(task="binary"),
        "recall": torchmetrics.Recall(task="binary"),
    })
    
    return metrics