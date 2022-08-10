from __future__ import absolute_import, division

import math
import random
import numbers
import warnings
from enum import IntEnum, Enum
from types import LambdaType
from typing import Optional, Union, Sequence, Tuple

import cv2
import numpy as np
from skimage.measure import label

from albumentations import (
    ImageOnlyTransform,
    to_tuple,
)


class UniformNoise(ImageOnlyTransform):
    """Apply gaussian noise to the input image.

    Args:
        magnitude (float): magnitude of uniform[-1; 1] noise
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, magnitude: float, always_apply=False, p=0.5):
        super(UniformNoise, self).__init__(always_apply, p)
        self.magnitude = magnitude

    def apply(self, img, uniform_noise=None, **params):
        return np.clip(img.astype(np.float32) + uniform_noise, 0, 255.)

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        uniform_noise = random_state.uniform(-1, 1, image.shape[:2]) * self.magnitude
        if len(image.shape) == 3:
            uniform_noise = np.expand_dims(uniform_noise, -1)

        return {"uniform_noise": uniform_noise}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("magnitude",)
