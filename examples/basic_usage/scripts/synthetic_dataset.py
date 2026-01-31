"""Synthetic Dataset for Basic Usage Example."""

import numpy as np
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from venturi import Config, instantiate


class SyntheticDataset:
    """Synthetic dataset containing squares and disks with noise."""

    def __init__(
        self,
        num_samples=100,
        num_channels=1,
        img_size=(64, 64),
        task="classification",
        transform=None,
        seed=42,
    ):
        """Args:
        num_samples: Total size of the dataset.
        num_channels: Number of image channels.
        img_size: (height, width).
        task: 'classification', 'segmentation', or 'regression'.
        seed: Base seed for deterministic generation.
        """
        self.num_samples = num_samples
        self.c = num_channels
        self.h, self.w = img_size
        self.task = task
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.RandomState(self.seed + idx)

        # 0 = Circle, 1 = Square
        shape_class = rng.randint(0, 2)

        size = rng.randint(int(self.w * 0.1), int(self.w * 0.3))
        cx = rng.randint(size, self.w - size)
        cy = rng.randint(size, self.h - size)

        y_grid, x_grid = np.meshgrid(np.arange(self.h), np.arange(self.w), indexing="ij")

        # Generate Binary Mask
        if shape_class == 0:
            dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            mask = (dist <= size).astype(np.float32)
        else:
            x_dist = np.abs(x_grid - cx)
            y_dist = np.abs(y_grid - cy)
            mask = ((x_dist <= size) & (y_dist <= size)).astype(np.float32)

        # Add channel dimension
        mask = mask[np.newaxis, :, :]

        intensity = rng.uniform(0.5, 1.0)
        # Create noise
        noise = rng.randn(self.c, self.h, self.w) * 0.1

        image = (mask * intensity) + noise
        image = np.clip(image, 0, 1).astype(np.float32)

        # Return based on Task
        if self.task == "classification":
            target = shape_class

        elif self.task == "segmentation":
            target = mask

        elif self.task == "regression":
            # Returns normalized coordinates [cx, cy, size]
            target = np.array([cx / self.w, cy / self.h, size / self.w], dtype=np.float32)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        if self.transform:
            image, target = self.transform((image, target))

        return image, target


class SegmentationTransform:
    """Transform pipeline for segmentation task."""

    def __init__(self, cfg_transforms: Config | None = None):
        """Args:
        cfg_transforms: Config containing the transforms to apply.
        """

        if cfg_transforms is None:
            transforms = v2.Identity()
        else:
            transforms_list = []
            for t in cfg_transforms.values():
                transforms_list.append(instantiate(t))
            transforms = v2.Compose(transforms_list)

        self.transforms = transforms

    def __call__(self, sample):
        """Applies the transform pipeline to the sample."""
        image, mask = sample

        img_tv = tv_tensors.Image(torch.from_numpy(image))
        mask_tv = tv_tensors.Mask(torch.from_numpy(mask))

        img_out, mask_out = self.transforms(img_tv, mask_tv)

        img_out = img_out.float()
        mask_out = mask_out.long()

        return img_out, mask_out


def get_segmentation_dataset(args: Config) -> tuple:
    """Creates train and validation SyntheticDataset for segmentation task."""

    args_d = args.dataset

    train_ds = SyntheticDataset(
        num_samples=args_d.params.num_train_samples,
        num_channels=args_d.params.num_channels,
        img_size=args_d.params.img_size,
        task="segmentation",
        transform=SegmentationTransform(args_d.train_transforms),
        seed=args.seed,
    )
    val_ds = SyntheticDataset(
        num_samples=args_d.params.num_val_samples,
        num_channels=args_d.params.num_channels,
        img_size=args_d.params.img_size,
        task="segmentation",
        transform=SegmentationTransform(args_d.val_transforms),
        seed=args.seed + 1,
    )

    return train_ds, val_ds
