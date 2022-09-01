import numpy as np

from albumentations import (
    ImageOnlyTransform,
)

from albumentations.augmentations import functional as F
from skimage.color import rgb2hed, hed2rgb


class ScikitPink(ImageOnlyTransform):
    """Randomly rearrange channels of the input RGB image.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    @property
    def targets_as_params(self):
        return ["image"]

    def apply(self, img, **params):
        hed = rgb2hed(img)
        hed = hed[..., (0, 2, 1)]
        rgb = hed2rgb(hed)
        rgb = rgb * 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def get_params_dependent_on_targets(self, params):
        return {}

    def get_transform_init_args_names(self):
        return ()