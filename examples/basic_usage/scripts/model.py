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


def get_model(args: Config) -> nn.Module:
    """Creates a simple CNN model based on the dataset parameters in args."""

    args_p = args.model.params

    layers = []

    layers.append(stage(args_p.num_input_channels, args_p.base_filters))
    for _ in range(args_p.num_layers):
        layers.extend([stage(args_p.base_filters, args_p.base_filters)])
    layers.append(stage(args_p.base_filters, args_p.num_output_channels))

    model = nn.Sequential(*layers)

    return model
