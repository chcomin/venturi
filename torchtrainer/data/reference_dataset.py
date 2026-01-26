import numpy as np
import torch
from sympy import Segment
from torchvision import tv_tensors
from torchvision.transforms import v2

from torchtrainer.engine.config import instantiate


class SyntheticDataset:
    """Synthetic dataset."""
    def __init__(
            self, 
            num_samples=100, 
            num_channels=1,
            img_size=(64, 64), 
            task="classification", 
            transform=None,
            seed=42
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
        
        y_grid, x_grid = np.meshgrid(
            np.arange(self.h), 
            np.arange(self.w), 
            indexing="ij"
        )
        
        # Generate Binary Mask
        if shape_class == 0: # Circle
            dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
            mask = (dist <= size).astype(np.float32)
        else: # Square
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
            target = np.array([cx/self.w, cy/self.h, size/self.w], dtype=np.float32)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        if self.transform:
            image, target = self.transform((image, target))
        
        return image, target

class BaseClassificationTransform:
    """Transform pipeline for classification task."""
    def __init__(self, cfg_transforms):

        if cfg_transforms is None:
            transforms = v2.Identity()
        else:
            transforms_list = []
            for t in cfg_transforms.values():
                transforms_list.append(instantiate(t))
            transforms = v2.Compose(transforms_list)

        self.transforms = transforms

    def __call__(self, sample):
        image, label = sample
        img_t = torch.from_numpy(image)
        label_t = torch.tensor(label, dtype=torch.long)
        img_out = self.transforms(img_t)

        img_out = img_out.float()

        return img_out, label_t

class BaseSegmentationTransform:
    """Transform pipeline for segmentation task."""
    def __init__(self, cfg_transforms):

        if cfg_transforms is None:
            transforms = v2.Identity()
        else:
            transforms_list = []
            for t in cfg_transforms.values():
                transforms_list.append(instantiate(t))
            transforms = v2.Compose(transforms_list)

        self.transforms = transforms

    def __call__(self, sample):
        image, mask = sample

        img_tv = tv_tensors.Image(torch.from_numpy(image))
        mask_tv = tv_tensors.Mask(torch.from_numpy(mask))

        img_out, mask_out = self.transforms(img_tv, mask_tv)

        img_out = img_out.float()
        mask_out = mask_out.long()

        return img_out, mask_out

class BaseRegressionTransform:
    """Transform pipeline for regression task."""
    def __init__(self, cfg_transforms):
        self.output_size = cfg_transforms.output_size
        
        transforms_list = []
        for t in cfg_transforms.values():
            if "_target_" in t:
                transforms_list.append(instantiate(t))
        if transforms_list == []:
            transforms = v2.Identity()
        else:
            transforms = v2.Compose(transforms_list)

        self.transforms = transforms

    def __call__(self, sample):
        image, target = sample # target is [cx, cy, radius] (normalized 0-1)
        h, w = image.shape[1], image.shape[2]

        # Convert normalized [cx, cy, r] to absolute pixels [x1, y1, x2, y2]
        cx, cy, r = target
        abs_cx = cx * w
        abs_cy = cy * h
        abs_r  = r * w # Assuming radius is relative to width
        
        box = [abs_cx - abs_r, abs_cy - abs_r, abs_cx + abs_r, abs_cy + abs_r]
        
        img_tv = tv_tensors.Image(torch.from_numpy(image))
        boxes_tv = tv_tensors.BoundingBoxes(            # type: ignore
            torch.tensor([box], dtype=torch.float32), 
            format="XYXY", 
            canvas_size=(h, w)
        )

        img_out, boxes_out = self.transforms(img_tv, boxes_tv)

        # Bounding Box -> Circle      
        if len(boxes_out) > 0:
            nx1, ny1, nx2, ny2 = boxes_out[0]
            
            new_w_px = nx2 - nx1
            new_h_px = ny2 - ny1
            
            out_cx_px = nx1 + (new_w_px / 2.0)
            out_cy_px = ny1 + (new_h_px / 2.0)
            out_r_px = (new_w_px + new_h_px) / 4.0
            out_h, out_w = self.output_size
            
            final_target = torch.tensor([
                out_cx_px / out_w, 
                out_cy_px / out_h, 
                out_r_px / out_w
            ], dtype=torch.float32)
        else:
            # Edge Case: The crop completely removed the object. Return zeros.
            final_target = torch.zeros(3, dtype=torch.float32)

        return img_out, final_target

def get_segmentation_dataset(stage, args):

    args_d = args.dataset

    if stage == "fit":
        seed = 0
        train_ds = SyntheticDataset(
            num_samples = args_d.params.num_train_samples,
            num_channels= args_d.params.num_channels,
            img_size = args_d.params.img_size,
            task = "segmentation",
            transform = BaseSegmentationTransform(args_d.train_transforms),
            seed = seed
        )
        val_ds = SyntheticDataset(
            num_samples = args_d.params.num_val_samples,
            num_channels= args_d.params.num_channels,
            img_size = args_d.params.img_size,
            task = "segmentation",
            transform = BaseSegmentationTransform(args_d.val_transforms),
            seed = seed + 1
        )
        return {"train_ds": train_ds, "val_ds": val_ds}
    elif stage == "test":
        seed = 2
    elif stage == "predict":
        seed = 3
    else:
        raise ValueError(f"Unknown stage: {stage}")

