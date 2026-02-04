import torch

class SimpleDataset(torch.utils.data.Dataset):
    """A mock dataset that generates random images and labels."""
    def __init__(self, num_classes, length=100, image_size=32):
        self.num_classes = num_classes
        self.length = length
        self.image_size = image_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        img = torch.randn(3, self.image_size, self.image_size)
        target = torch.randint(0, self.num_classes, (1,)).item()

        return img, target


def get_dataset(vcfg):

    return {
        "train_ds": SimpleDataset(vcfg.dataset.num_classes), 
        "val_ds": SimpleDataset(vcfg.dataset.num_classes)
        }
