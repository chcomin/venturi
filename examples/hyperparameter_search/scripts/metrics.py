"""Losses and metrics for binary segmentation tasks."""

import torchmetrics

from venturi import Config


def classification_metrics(vcfg: Config) -> torchmetrics.MetricCollection:
    """Metrics for classification.

    Args:
        vcfg: Configuration object containing hyperparameters.
    """

    num_classes = vcfg.dataset.num_classes

    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes),
            "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes),
        }
    )

    return metrics
