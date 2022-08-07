import albumentations as A
import cv2

from albumentations import (
    Resize,
    Normalize,
    Compose, Transpose,
    HorizontalFlip,
    VerticalFlip, Rotate, RandomRotate90,
    ShiftScaleRotate, ElasticTransform, Affine,
    GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
    RandomBrightnessContrast, HueSaturationValue, IAASharpen,
    RandomGamma, RandomBrightness, RandomBrightnessContrast,
    GaussianBlur, CLAHE, RGBShift, RandomResizedCrop,
    Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
    Normalize, OneOf, NoOp
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional
from .extra_augs import RandStainNA


def get_test_augmentations() -> Compose:
    return Compose(
        [
            Normalize(),
            ToTensorV2()
        ]
    )

# 1024 - 1024

def get_simple_augmentations(
        train: bool = True,
        height: int = 512,
        width: int = 512
    ) -> Compose:
    if train:
        return Compose(
            [
                #RandomCrop(height, width, p=1.0),
                #Morphology
                RandomResizedCrop(
                    height=height,
                    width=width,
                    scale=(0.4, 0.6),
                    interpolation=cv2.INTER_LANCZOS4,
                    always_apply=True
                ),
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Transpose(p=0.5),

                ShiftScaleRotate(shift_limit=0, scale_limit=0.01,
                                 rotate_limit=(-45,45),
                                 interpolation=cv2.INTER_LANCZOS4,
                                 border_mode=0, p=0.5),


                #RandStainNA(p=1.0),

                # NEW
                OneOf(
                    [
                        GaussianBlur(blur_limit=(3,5), p=0.5),
                        GaussNoise(var_limit=(25, 30), p=0.5)
                    ],
                    p=0.5
                ),
                ChannelShuffle(p=0.5),
                OpticalDistortion(
                    p=0.5,
                    interpolation=cv2.INTER_LANCZOS4
                ),
                Affine(
                    p=0.5,
                    interpolation=cv2.INTER_LANCZOS4
                ),
                #Color
                OneOf(
                    [
                        CLAHE(p=0.5), #NEW
                        RGBShift(p=0.5),
                        RandomGamma(p=0.5),
                        RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                                 brightness_by_max=True,p=0.5),
                        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                                           val_shift_limit=20, p=0.5),
                    ],
                    p=0.875
                ),
                CoarseDropout(max_holes=2,
                              max_height=height//4, max_width=width//4,
                              min_holes=1,
                              min_height=height//16, min_width=width//16,
                              fill_value=0, mask_fill_value=0, p=0.5),
                Normalize(),
                ToTensorV2()
            ]
        )
    else:
        # INTER_LANCZOS4
        return Compose(
            [
                Normalize(),
                ToTensorV2()
            ]
        )



