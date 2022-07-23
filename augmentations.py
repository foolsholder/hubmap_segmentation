from albumentations import (
    Resize,
    Normalize,
    Compose,
    HorizontalFlip,
    VerticalFlip, Rotate, RandomRotate90,
    ShiftScaleRotate, ElasticTransform,
    GridDistortion, RandomSizedCrop, RandomCrop, CenterCrop,
    RandomBrightnessContrast, HueSaturationValue, IAASharpen,
    RandomGamma, RandomBrightness, RandomBrightnessContrast,
    GaussianBlur,CLAHE,
    Cutout, CoarseDropout, GaussNoise, ChannelShuffle, ToGray, OpticalDistortion,
    Normalize, OneOf, NoOp
)
from albumentations.pytorch import ToTensorV2
from typing import Union, Dict, Any, Optional


_HEIGHT = 512
_WIDTH = 512


def get_simple_augmentations(train: bool = True) -> Compose:
    if train:
        return Compose(
            [
                RandomRotate90(p=1),
                HorizontalFlip(p=0.5),

                #Morphology
                ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30),
                                 interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
                GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
                GaussianBlur(blur_limit=(3,7), p=0.5),

                #Color
                RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                         brightness_by_max=True,p=0.5),
                HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                                   val_shift_limit=0, p=0.5),

                CoarseDropout(max_holes=2,
                              max_height=_HEIGHT//4, max_width=_WIDTH//4,
                              min_holes=1,
                              min_height=_HEIGHT//16, min_width=_WIDTH//16,
                              fill_value=0, mask_fill_value=0, p=0.5),
                Normalize(),
                ToTensorV2()
            ]
        )
    else:
        return Compose(
            [
                Normalize(),
                ToTensorV2()
            ]
        )



