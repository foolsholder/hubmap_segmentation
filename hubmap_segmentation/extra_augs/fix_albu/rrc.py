import math
import random
import PIL
import numpy as np
import cv2

from albumentations.augmentations.crops import functional as F
from albumentations.augmentations.crops import RandomResizedCrop


class FRandomResizedCrop(RandomResizedCrop):
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        max_elem = np.max(img)
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = self.apply(img, **{k: cv2.INTER_LINEAR if k == "interpolation" else v for k, v in params.items()})
        augmented_mask = (augmented_mask > 0).astype(img.dtype) * max_elem
        return augmented_mask
