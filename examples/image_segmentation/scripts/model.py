"""Simple example of a model creation function."""

from torch import nn

from venturi import Config


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Creates a simple stage with Conv2d, BatchNorm2d, and ReLU layers."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class SimpleCNN(nn.Module):
    """A simple CNN model for image segmentation."""

    def __init__(self, in_channels: int, out_channels: int, base_filters: int, num_layers: int):
        """Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_filters: Number of filters in the first layer.
        num_layers: Number of convolutional layers.
        """
        super().__init__()

        layers = []

        layers.append(conv_block(in_channels, base_filters))
        for _ in range(num_layers):
            layers.extend([conv_block(base_filters, base_filters)])
        layers.append(conv_block(base_filters, out_channels))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)


def get_model(vcfg: Config) -> nn.Module:
    """Creates a simple CNN model based on the dataset parameters in vcfg."""

    vcfg_p = vcfg.model.params

    model = SimpleCNN(
        vcfg_p.num_input_channels,
        vcfg_p.num_output_channels,
        vcfg_p.base_filters,
        vcfg_p.num_layers,
    )

    return model
