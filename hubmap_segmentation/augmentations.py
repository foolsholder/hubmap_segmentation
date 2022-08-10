import albumentations as A
import cv2

from albumentations import (
    Resize, Normalize, Compose, Transpose,
    HorizontalFlip, VerticalFlip, Rotate, RandomRotate90,
    ShiftScaleRotate, ElasticTransform, Affine, PadIfNeeded,
    GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
    RandomBrightnessContrast, HueSaturationValue, IAASharpen,
    RandomGamma, RandomBrightness, RandomBrightnessContrast,
    GaussianBlur, CLAHE, RGBShift, RandomResizedCrop, ImageCompression,
    Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
    Normalize, OneOf, NoOp
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional
from .extra_augs import RandStainNA, UniformNoise


def get_norm_tensor_augmentations() -> Compose:
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
                RandomCrop(
                    height=height,
                    width=width,
                    always_apply=True
                ), # 1024x1024 -> 512x512
                RandomRotate90(p=0.5),
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                Transpose(p=0.5),

                #RandStainNA(p=1.0),
                ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
                # NEW
                ChannelShuffle(p=0.5),
                RGBShift(p=0.5),
                CLAHE(p=0.5), #NEW
                OneOf([
                    RandomGamma(p=0.5),
                    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4,
                                             brightness_by_max=True,p=0.5)
                    ],
                    p=0.5
                ),
                HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40,
                                   val_shift_limit=25, p=0.5),

                #OneOf(
                #    [
                #        OpticalDistortion(
                #            interpolation=cv2.INTER_LANCZOS4,
                #            p=0.5
                #        ),
                #        GridDistortion(
                #            interpolation=cv2.INTER_LANCZOS4,
                #            p=0.5
                #        ),
          #              ElasticTransform(
          #                  alpha_affine=15,
          #                  sigma=1,
          #                  interpolation=cv2.INTER_LANCZOS4,
          #                  p=0.5
          #              ),
                #    ],
                #    p=0.5
                #),

                OneOf(
                    [
                        GaussianBlur(blur_limit=(7, 19), p=0.35),
                        GaussNoise(var_limit=30, p=0.5, per_channel=False),
                        UniformNoise(magnitude=25, p=0.5),
                    ],
                    p=0.5,
                ),

                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                 rotate_limit=(-45,45),
                                 interpolation=cv2.INTER_LANCZOS4,
                                 border_mode=0, p=0.5),


                Normalize(),
                ToTensorV2()
            ]
        )
    else:
        # INTER_LANCZOS4
        return Compose(
            [
                #Resize(
                #    height=height,
                #    width=width,
                #    interpolation=cv2.INTER_LANCZOS4
                #),
                Normalize(),
                ToTensorV2()
            ]
        )



