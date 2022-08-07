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
from .augmentations import get_simple_augmentations, get_test_augmentations


class SDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            test: bool = False,
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
        if train:
            self.folder_images = 'tiled_images/images_{}'.format(height * 2)
            self.folder_masks = 'tiled_images/masks_{}'.format(height * 2)
        elif not test:
            self.folder_images = 'tiled_images/images_{}'.format(height)
            self.folder_masks = 'tiled_images/masks_{}'.format(height)
        else:
            self.folder_images = 'full_images'
            self.folder_masks = 'full_masks'

        suffix_csv = 'tiled_train' if train else 'tiled_valid'
        if test or fold is None:
            test = True
            suffix_csv = 'test'
        else:
            suffix_csv += '_' + str(fold)

        self.test = test
        self.test_aug = get_test_augmentations()
        self.df = pd.read_csv(os.path.join(
            root,
            'csv_files',
            suffix_csv + '.csv'
        ))
        #self.df = self.df[self.df.organ != 'kidney']
        self.organ2id = {
            'kidney': 0,
            'largeintestine': 1,
            'lung': 2,
            'prostate': 3,
            'spleen': 4
        }
        if augs is None:
            augs = get_simple_augmentations(train, height=height, width=width)
        self.augs = augs

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
            self,
            index: int
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        index = index % len(self.df)
        row = self.df.iloc[index]
        image_id = str(row['id'])
        image = np.load(os.path.join(
            self.root,
            self.folder_images,
            image_id + '.npy'
        ))
        target = np.load(os.path.join(
            self.root,
            self.folder_masks,
            image_id + '.npy'
        ))
        aug_dict = self.get_augmented(image=image, mask=target)
        res = {
            "input_x": aug_dict['image'],
            "target": aug_dict['mask'][None],
            'image_id': image_id,
            'organ': row['organ'],
            'organ_id': self.organ2id[row['organ']]
        }
        if not self.train:
            res.update({
                'full_target': torch.Tensor(target)[None],
            })
        if self.test:
            res.update({
                'full_image': self.test_aug(image=image)['image'],
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
        test: bool = False,
        batch_size: int = 4,
        num_workers: int = 4,
        height: int = 512,
        width: int = 512,
        fold: Optional[int] = None
    ) -> DataLoader:
    dataset = SDataset(train, test, height=height, width=width, fold=fold)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train
    )


def create_loader_from_cfg(cfg_loader: Dict[str, Any]) -> DataLoader:
    return create_loader(**cfg_loader)
