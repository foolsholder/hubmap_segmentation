import numpy as np
import torchvision.transforms

import cv2
import random
from albumentations.core.transforms_interface import to_tuple
from albumentations.augmentations.geometric import functional as FGEOM
from albumentations.augmentations import functional as F
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, ElasticTransform
from albumentations.augmentations.transforms import GridDistortion, OpticalDistortion


class FShiftScaleRotate(ShiftScaleRotate):
    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        max_elem = np.max(img) # dtype == int
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = FGEOM.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_LINEAR, self.border_mode, self.mask_value)
        augmented_mask = (augmented_mask > 0.2).astype(img.dtype) * max_elem
        return augmented_mask


class FElasticTransform(ElasticTransform):
    def apply_to_mask(self, img, random_state=None, **params):
        max_elem = np.max(img)  # dtype == int
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = FGEOM.elastic_transform(
            img,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_LINEAR,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
            self.same_dxdy,
        )
        augmented_mask = (augmented_mask > 0.2).astype(img.dtype) * max_elem
        return augmented_mask


class FOpticalDistortion(OpticalDistortion):
    def apply_to_mask(self, img, k=0, dx=0, dy=0, **params):
        max_elem = np.max(img)  # dtype == int
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = F.optical_distortion(img, k, dx, dy, cv2.INTER_LINEAR, self.border_mode, self.mask_value)
        augmented_mask = (augmented_mask > 0.2).astype(img.dtype) * max_elem
        return augmented_mask


class FGridDistortion(GridDistortion):
    def apply_to_mask(self, img, stepsx=(), stepsy=(), **params):
        max_elem = np.max(img)  # dtype == int
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = F.grid_distortion(
            img,
            self.num_steps,
            stepsx,
            stepsy,
            cv2.INTER_LINEAR,
            self.border_mode,
            self.mask_value,
        )
        augmented_mask = (augmented_mask > 0.2).astype(img.dtype) * max_elem
        return augmented_mask
