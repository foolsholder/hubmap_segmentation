import random
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2


from albumentations.core.transforms_interface import to_tuple


from .dual import FDualTransform
from albumentations.augmentations import functional as F


class FShiftScaleRotate(FDualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        shift_limit_x ((float, float) or float): shift factor range for width. If it is set then this value
            instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
            the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
            the range [0, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=45,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        shift_limit_x=None,
        shift_limit_y=None,
        rotate_method="largest_box",
        always_apply=False,
        p=0.5,
    ):
        super(FShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method

        if self.rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return F.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_NEAREST_EXACT, self.border_mode, self.mask_value)

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, **params):
        return F.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)

    def get_params(self):
        return {
            "angle": random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
            "scale": random.uniform(self.scale_limit[0], self.scale_limit[1]),
            "dx": random.uniform(self.shift_limit_x[0], self.shift_limit_x[1]),
            "dy": random.uniform(self.shift_limit_y[0], self.shift_limit_y[1]),
        }

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, **params):
        return F.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, self.rotate_method, **params)

    def get_transform_init_args(self):
        return {
            "shift_limit_x": self.shift_limit_x,
            "shift_limit_y": self.shift_limit_y,
            "scale_limit": to_tuple(self.scale_limit, bias=-1.0),
            "rotate_limit": self.rotate_limit,
            "interpolation": self.interpolation,
            "border_mode": self.border_mode,
            "value": self.value,
            "mask_value": self.mask_value,
            "rotate_method": self.rotate_method,
        }

