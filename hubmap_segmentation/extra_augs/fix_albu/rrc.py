import math
import random
import PIL

from albumentations.augmentations.crops import functional as F
from .dual import FDualTransform
from .functional import resize


class _FBaseRandomSizedCrop(FDualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, height, width, interpolation=PIL.Image.Resampling.BILINEAR, always_apply=False, p=1.0):
        super(_FBaseRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=PIL.Image.Resampling.BILINEAR, **params):
        crop = F.random_crop(img, crop_height, crop_width, h_start, w_start)
        #return FGeometric.resize(crop, self.height, self.width, interpolation)
        return resize(crop, self.height, self.width, interpolation)


class FRandomResizedCrop(_FBaseRandomSizedCrop):

    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=PIL.Image.Resampling.BILINEAR,
        always_apply=False,
        p=1.0,
    ):

        super(FRandomResizedCrop, self).__init__(
            height=height, width=width, interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "height", "width", "scale", "ratio", "interpolation"