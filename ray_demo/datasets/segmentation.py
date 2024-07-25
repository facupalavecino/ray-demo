import os
import numpy as np

from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_files: List[str],
        mask_files: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]

        image = np.load(os.path.join(self.images_dir, image_file))
        mask = np.expand_dims(
            np.load(os.path.join(self.masks_dir, mask_file)),
            axis=2
        )

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
