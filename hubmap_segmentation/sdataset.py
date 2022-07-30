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
from .augmentations import get_simple_augmentations


class SDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            root: str = '',
            augs: Optional[Compose] = None,
            height: int = 512,
            width: int = 512,
            fold: Optional[int] = None
    ):
        super(SDataset, self).__init__()
        if root == '':
            root = os.environ['SDATASET_PATH']
        self.height = height
        self.width = width
        self.train = train
        self.data_folder: str = root
        self.root = root
        suffix_csv = 'train' if train else 'valid'
        self.suffix = suffix_csv
        fold_str: str = ('_' + str(fold)) if fold is not None else ''
        self.df = pd.read_csv(os.path.join(
            root,
            'csv_files',
            suffix_csv + fold_str + '.csv'
        ))
        if augs is None:
            augs = get_simple_augmentations(train, height=height, width=width)
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
            'images_{}'.format(self.height),
            image_id + '.npy'
        ))
        # image currently in BGR format
        target = np.load(os.path.join(
            self.root,
            'resized_images',
            'masks_{}'.format(self.height),
            image_id + '.npy'
        ))
        aug_dict = self.get_augmented(image=image, mask=target)
        res = {
            "input_x": aug_dict['image'],
            "target": aug_dict['mask'][None]
        }
        if not self.train:
            full_target = np.load(os.path.join(
                self.root,
                'full_masks',
                image_id + '.npy'
            ))
            res.update({
                'full_target': torch.Tensor(full_target)[None],
                'image_id': image_id,
                'organ': row['organ']
            })
        return res

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
        num_workers: int = 4,
        height: int = 512,
        width: int = 512,
        fold: Optional[int] = None
    ) -> DataLoader:
    dataset = SDataset(train, height=height, width=width, fold=fold)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train
    )


def create_loader_from_cfg(cfg_loader: Dict[str, Any]) -> DataLoader:
    return create_loader(**cfg_loader)
