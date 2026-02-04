"""Defines simple CNN and Vision Transformer models for hyperparameter search examples."""

from torch import nn
from torchvision.models import VisionTransformer


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """A simple convolutional block: Conv2d -> BatchNorm2d -> ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class SimpleCNN(nn.Module):
    """A simple CNN model with configurable depth and number of filters."""
    def __init__(self, in_channels=3, num_classes=10, num_layers=2, num_filters=16):
        """Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        num_classes (int): Number of output classes.
        num_layers (int): Number of convolutional layers.
        num_filters (int): Number of filters in each convolutional layer.
        """
        super().__init__()

        layers = []
        layers.append(
            conv_block(in_channels, num_filters, kernel_size=3, padding=1)
        )
        for _ in range(1, num_layers):
            layers.append(
                conv_block(num_filters, num_filters, kernel_size=3, padding=1)
            )
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, num_classes)
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the model."""
        return self.layers(x)
    
class SimpleViT(nn.Module):
    """A simple Vision Transformer model with configurable depth and number of heads."""
    def __init__(
            self, 
            num_classes=10,
            num_layers=6, 
            hidden_dim=16,   
            num_heads=2,
            image_size=32, 
            patch_size=8, 
            ):
        """Args:
        num_classes: Number of output classes.
        num_layers: Number of transformer layers.
        hidden_dim: Dimension of the hidden layers.
        num_heads: Number of attention heads.
        image_size: Size of the input images (assumed square).
        patch_size: Size of the patches (assumed square).
        """
        super().__init__()        

        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=hidden_dim * 2,
            num_classes=num_classes,
        )

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
def get_cnn(vcfg):
    """Instantiate a SimpleCNN model based on the provided configuration."""
    return SimpleCNN(
        num_layers=vcfg.model.num_layers,
        num_filters=vcfg.model.num_filters,
    )

    
def get_vit(vcfg):
    """Instantiate a Vision Transformer model based on the provided configuration."""
    return SimpleViT(
        num_layers=vcfg.model.num_layers,
        hidden_dim=vcfg.model.hidden_dim,
    )