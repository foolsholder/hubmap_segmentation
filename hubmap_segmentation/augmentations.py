import albumentations as A
import cv2
import PIL
import torchvision

from albumentations import (
    Resize, Normalize, Compose, Transpose,
    HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
    ElasticTransform, Affine, PadIfNeeded,
    GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
    RandomBrightnessContrast, HueSaturationValue, IAASharpen,
    RandomGamma, RandomBrightness, RandomBrightnessContrast,
    GaussianBlur, CLAHE, RGBShift, ImageCompression,
    Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
    Normalize, OneOf, NoOp, ShiftScaleRotate, RandomResizedCrop
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional
from .extra_augs import RandStainNA, UniformNoise, ScikitPink


from .extra_augs.fix_albu import (
    FRandomResizedCrop, FShiftScaleRotate, FOpticalDistortion,
    FGridDistortion, FElasticTransform
)


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
                #FRandomResizedCrop(
                #    height=height,
                #    width=width,
                #    interpolation=cv2.INTER_LANCZOS4,
                #    scale=(0.20, 0.30),  # mean - 0.25
                #    always_apply=True
                #),
                RandomCrop(
                    height=height,
                    width=width,
                    always_apply=True
                ), # 1024x1024 -> 512x512
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Transpose(p=0.5),

                ScikitPink(p=0.5),
                ChannelShuffle(p=0.5),

                CLAHE(p=0.75),
                RGBShift(p=0.75),
                ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
                # NEW
                OneOf([
                    RandomGamma(p=0.15),
                    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4,
                                             brightness_by_max=True, p=0.15),
                    ],
                    p=0.5
                ),
                HueSaturationValue(hue_shift_limit=60, sat_shift_limit=50,
                                   val_shift_limit=50, p=0.8),
                OneOf(
                    [
                        #OpticalDistortion(interpolation=cv2.INTER_LANCZOS4, border_mode=0, p=0.3),
                        ElasticTransform(alpha_affine=15, sigma=7, interpolation=cv2.INTER_LANCZOS4, border_mode=0, p=0.6),
                        #GridDistortion(distort_limit=0.1, interpolation=cv2.INTER_LANCZOS4, border_mode=0, p=0.3)
                    ],
                    p=0.5
                ),


                OneOf(
                    [
                        GaussianBlur(blur_limit=(5, 13), p=0.5),
                        GaussNoise(var_limit=20, p=0.5, per_channel=True),
                    ],
                    p=0.5,
                ),
                ShiftScaleRotate(
                    rotate_limit=45,
                    shift_limit=0.15,
                    scale_limit=0.2,
                    interpolation=cv2.INTER_LANCZOS4,
                    border_mode=0,
                    p=0.7
                ),

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