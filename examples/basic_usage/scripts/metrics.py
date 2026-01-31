"""Losses and metrics for binary segmentation tasks."""

import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import BinaryF1Score

from venturi import Config


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    """Similar to nn.BCEWithLogitsLoss with pos_weight, but matches exactly the behavior of
    nn.CrossEntropyLoss when classes are weighted.
    """

    def __init__(self, weight: tuple | None = None, reduction="mean"):
        """Args:
        weight: An optional tuple containing weights for the negative and positive examples.
        Useful for unbalanced datasets.
        reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        if weight is None:
            weight = torch.tensor([1.0, 1.0])
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        # Ensure weight is a tensor and registers it as a buffer so it moves to GPU automatically
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        # Move weight to correct device if needed
        if self.weight.device != input.device:
            self.weight = self.weight.to(input.device)

        target = target.float()

        losses = F.binary_cross_entropy_with_logits(input, target, reduction="none")
        sample_weights = target * self.weight[1] + (1 - target) * self.weight[0]
        # Apply the weights to the losses
        weighted_losses = losses * sample_weights

        # Perform the reduction (Exact CrossEntropyLoss behavior)
        if self.reduction == "mean":
            return weighted_losses.sum() / sample_weights.sum()
        elif self.reduction == "sum":
            return weighted_losses.sum()
        else:
            return weighted_losses


class Dice(BinaryF1Score):
    """Dice score for binary segmentation."""

    def __init__(self, threshold=0.5, ignore_index=None, **kwargs):
        """Args:
        threshold: Threshold for converting predicted probabilities to binary (0 or 1).
        ignore_index: Class index to ignore when computing the metric.
        **kwargs: Additional keyword arguments for torchmetrics.classification.BinaryF1Score.
        """
        # 'samplewise' returns Dice per image
        super().__init__(
            threshold=threshold, multidim_average="samplewise", ignore_index=ignore_index, **kwargs
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update state with predictions and targets."""
        if target.is_floating_point():
            target = target.long()
        super().update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute mean Dice score over the batch."""
        val = super().compute()
        return val.mean()


def binary_segmentation_metrics(args: Config) -> torchmetrics.MetricCollection:
    """Define common metrics for binary segmentation tasks.

    Args:
        args (Config): Configuration object containing hyperparameters.
    """

    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="binary"),
            "dice": Dice(),
            "precision": torchmetrics.Precision(task="binary"),
            "recall": torchmetrics.Recall(task="binary"),
        }
    )

    return metrics
