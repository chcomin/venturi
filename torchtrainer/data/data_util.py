"""Utility functions for data manipulation."""

from pathlib import Path

from torch.utils.data import Dataset


class Subset(Dataset):
    """Create a new Dataset containing a subset of images from the input Dataset."""

    def __init__(self, ds, indices, transforms=None, **attributes):
        """Args:
        ds : input Dataset
        indices: indices to use for the new dataset
        transforms: transformations to apply to the data. Defaults to None.
        attributes: additional attributes to store in the new dataset.
        """

        self.ds = ds
        self.indices = indices
        self.transforms = transforms
        for k, v in attributes.items():
            setattr(self, k, v)

    def __getitem__(self, idx):

        img, target = self.ds[self.indices[idx]]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)


def search_files(files: list[str], paths: list[Path]) -> list[int]:
    """Search for each file in list `paths` and return the indices of the files found
    in the paths list. Paths can then be fitlered as:

    >>> paths = [paths[idx] for idx in search_files(files, paths)] 

    Parameters
    ----------
    files
        List of file names to search   
    paths   
        List of paths.

    Returns:
    -------
    List of indices of the files found in the paths list
    """

    indices = []
    for file in files:
        for idx, path in enumerate(paths):
            if file in str(path):
                indices.append(idx)

    return indices
        