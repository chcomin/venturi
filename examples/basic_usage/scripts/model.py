"""Simple example of a model creation function."""

from torch import nn

from venturi import Config


def stage(in_channels: int, out_channels: int) -> nn.Sequential:
    """Creates a simple stage with Conv2d, BatchNorm2d, and ReLU layers."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def get_model(vcfg: Config) -> nn.Module:
    """Creates a simple CNN model based on the dataset parameters in vcfg."""

    vcfg_p = vcfg.model.params

    layers = []

    layers.append(stage(vcfg_p.num_input_channels, vcfg_p.base_filters))
    for _ in range(vcfg_p.num_layers):
        layers.extend([stage(vcfg_p.base_filters, vcfg_p.base_filters)])
    layers.append(stage(vcfg_p.base_filters, vcfg_p.num_output_channels))

    model = nn.Sequential(*layers)

    return model
