import io
import boto3
import os
import numpy as np

from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional


class SegmentationS3Dataset(Dataset):
    def __init__(
        self,
        bucket: str,
        image_files: List[str],
        mask_files: List[str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.bucket = bucket
        self.image_files = image_files
        self.mask_files = mask_files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index) -> Any:
        s3 = boto3.client("s3")
        image_file = self.image_files[index]
        mask_file = self.mask_files[index]

        image_obj = s3.get_object(Bucket=self.bucket, Key=image_file)
        image = np.load(io.BytesIO(image_obj["Body"].read()))

        mask_obj = s3.get_object(Bucket=self.bucket, Key=mask_file)
        mask = np.expand_dims(np.load(io.BytesIO(mask_obj["Body"].read())), axis=2)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
