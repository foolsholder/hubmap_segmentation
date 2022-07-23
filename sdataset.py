import torch
import os
import cv2

import albumentations as A
import numpy as np
import pandas as pd


from torch.utils.data import Dataset, DataLoader
from albumentations import (
    Resize,
    Normalize,
    Compose,
    HorizontalFlip
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional
from augmentations import get_simple_augmentations

_HEIGHT = 512
_WIDTH = 512


class SDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            root: str = '',
            augs: Optional[Compose] = None
    ):
        super(SDataset, self).__init__()
        if root == '':
            root = os.environ['SDATASET_PATH']
        self.train = train
        self.data_folder: str = root
        self.root = root
        suffix_csv = 'train' if train else 'valid'
        self.suffix = suffix_csv
        self.df = pd.read_csv(os.path.join(
            root,
            'csv_files',
            suffix_csv + '.csv'
        ))
        if augs is None:
            augs = get_simple_augmentations(train)
        self.augs = augs

    def __len__(self) -> int:
        return len(self.df) * (10 if self.train else 1)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        index = index % len(self.df)
        row = self.df.iloc[index]
        image_id = str(row['id'])
        image = np.load(os.path.join(
            self.root,
            'resized_images',
            'images_{}'.format(_HEIGHT),
            image_id + '.npy'
        ))
        # image currently in BGR format
        target = np.load(os.path.join(
            self.root,
            'resized_images',
            'masks_{}'.format(_HEIGHT),
            image_id + '.npy'
        ))
        aug_dict = self.get_augmented(image=image, mask=target)
        return {
            "input_x": aug_dict['image'],
            "target": aug_dict['mask'][None]
        }

    def get_augmented(
            self,
            image: np.array,
            mask: np.array
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        aug = self.augs(image=image, mask=mask)
        while aug['mask'].sum() < 1.0:
            aug = self.augs(image=image, mask=mask)
        return aug

def create_loader(
        train: bool = True,
        batch_size: int = 4,
        num_workers: int = 4
    ) -> DataLoader:
    dataset = SDataset(train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train
    )
