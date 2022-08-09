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
from .augmentations import get_simple_augmentations, get_norm_tensor_augmentations


class SDataset(Dataset):
    def __init__(
            self,
            train: bool = True,
            root: str = '',
            augs: Optional[Compose] = None,
            height: int = 512,
            width: int = 512,
            fold: Optional[int] = None,
            num_classes: int = 1
    ):
        super(SDataset, self).__init__()
        self.num_classes = num_classes
        if root == '':
            root = os.environ['SDATASET_PATH']
        self.height = height
        self.width = width
        self.train = train
        self.data_folder: str = root
        self.root = root

        self.folder_images = 'resized_images/images_scaled_{}'.format(2 * height)
        self.folder_masks = 'resized_images/masks_scaled_{}'.format(2 * height)

        suffix_csv = 'train_{}' if train else 'valid_{}'
        suffix_csv = suffix_csv.format(fold)

        self.df = pd.read_csv(os.path.join(
            root,
            'csv_files',
            suffix_csv + '.csv'
        ))
        #self.df = self.df[self.df.organ != 'kidney']
        self.organ2id = {
            'kidney': 1,
            'largeintestine': 2,
            'lung': 3,
            'prostate': 4,
            'spleen': 5
        }
        if augs is None:
            augs = get_simple_augmentations(train, height=height, width=width)
        self.norm_tensor = get_norm_tensor_augmentations()
        self.augs = augs

    def __len__(self) -> int:
        return len(self.df) * (5 if self.train else 1)

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
        ))#.astype(np.float32)
        #image = (image / np.max(image) * 255).astype(np.uint8)
        target = np.load(os.path.join(
            self.root,
            self.folder_masks,
            image_id + '.npy'
        ))
        aug_dict = self.get_augmented(image=image, mask=target)
        mask = aug_dict['mask'][None]
        res = {
            "input_x": aug_dict['image'],
            "target": self._to_ohe(mask, self.organ2id[row['organ']]),
            'image_id': image_id,
            'organ': row['organ'],
            'organ_id': self.organ2id[row['organ']]
        }
        if not self.train:
            full_mask = np.load(os.path.join(
                self.root,
                'full_masks',
                image_id + '.npy'
            ))
            full_mask = torch.Tensor(full_mask)[None]
            res.update({
                'full_target': full_mask,
                'full_image': self.norm_tensor(image=image)['image']
            })
        return res

    def _to_ohe(self, mask_tensor: torch.Tensor, organ_id: int) -> torch.Tensor:
        if self.num_classes == 1:
            return mask_tensor
        zero_cls = 1 - mask_tensor
        arr = [zero_cls]
        arr += [torch.zeros_like(mask_tensor)] * (organ_id - 1)
        arr += [mask_tensor]
        arr += [torch.zeros_like(mask_tensor)] * (self.num_classes - organ_id - 1)
        return torch.cat(arr, dim=0) # [NUM_CLASSES; H; W]

    def get_augmented(
            self,
            image: np.array,
            mask: np.array
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        aug = self.augs(image=image, mask=mask)
        #while aug['mask'].sum() < 1.0:
        #    aug = self.augs(image=image, mask=mask)
        #    if np.random.rand() < self.prob_miss:
        #        break
        return aug


def create_loader(
        train: bool = True,
        batch_size: int = 4,
        num_workers: int = 4,
        height: int = 512,
        width: int = 512,
        fold: Optional[int] = None,
        num_classes: int = 1
    ) -> DataLoader:
    dataset = SDataset(train,
                       height=height, width=width,
                       fold=fold, num_classes=num_classes)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=train,
        drop_last=train
    )


def create_loader_from_cfg(cfg_loader: Dict[str, Any]) -> DataLoader:
    return create_loader(**cfg_loader)
