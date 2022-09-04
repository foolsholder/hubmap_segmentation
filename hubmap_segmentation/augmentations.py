import albumentations as A
import cv2

from albumentations import (
    Resize, Normalize, Compose, Transpose,
    HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
    ElasticTransform, Affine, PadIfNeeded,
    GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
    RandomBrightnessContrast, HueSaturationValue, IAASharpen,
    RandomGamma, RandomBrightness, RandomBrightnessContrast,
    GaussianBlur, CLAHE, RGBShift, ImageCompression,
    Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
    Normalize, OneOf, NoOp
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional
from .extra_augs import RandStainNA, UniformNoise, ScikitPink


from .extra_augs.fix_albu import FRandomResizedCrop, FShiftScaleRotate


def get_norm_tensor_augmentations() -> Compose:
    return Compose(
        [
            #PadIfNeeded(
            #    min_height=1024,
            #    min_width=1024,
            #    border_mode=0,
            #),
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
                FRandomResizedCrop(
                    height=height,
                    width=width,
                    interpolation=cv2.INTER_LANCZOS4,
                    scale=(0.20, 0.30),  # mean - 0.25
                    always_apply=True
                ),
                #RandomCrop(
                #    height=height,
                #    width=width,
                #    always_apply=True
                #), # 1024x1024 -> 512x512
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Transpose(p=0.5),

                #ScikitPink(p=0.5),

                ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
                # NEW
                ChannelShuffle(p=0.75),
                RGBShift(p=0.75),
                CLAHE(p=0.75), #NEW
                OneOf([
                    RandomGamma(p=0.5),
                    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4,
                                             brightness_by_max=True,p=0.5),
                    ],
                    p=0.75
                ),
                HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40,
                                   val_shift_limit=25, p=0.75),


                OneOf(
                    [
                        GaussianBlur(blur_limit=(7, 19), p=0.5),
                        GaussNoise(var_limit=30, p=0.5, per_channel=True),
                    ],
                    p=0.5,
                ),
                FShiftScaleRotate(shift_limit=0.15, scale_limit=0.2,
                                 rotate_limit=(-45, 45),
                                 interpolation=cv2.INTER_LANCZOS4,
                                 border_mode=0, p=0.7),

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
