import numpy as np
import torchvision.transforms

from .functional import TorchAffine
from .dual import FDualTransform
from PIL import Image


class FShiftScaleRotate(FDualTransform):

    def __init__(
        self,
        width=512,
        height=512,
        always_apply=False,
        p=0.5,
        **torch_params
    ):
        super(FShiftScaleRotate, self).__init__(always_apply, p)
        self.torch_affine = TorchAffine(**torch_params)
        self.img_size = (width, height)

    def apply(self, img, affine_params, interpolation, **params):
        img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(np.uint8))
        img = self.torch_affine.right_forward(img, interpolation, affine_params)
        return np.array(img)
        #return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, **params):
        return self.apply(img, **{k: torchvision.transforms.InterpolationMode.NEAREST if k == "interpolation" else v for k, v in params.items()})

    def get_params(self):
        params = self.torch_affine.get_params(self.torch_affine.degrees, self.torch_affine.translate,
                                            self.torch_affine.scale, self.torch_affine.shear, self.img_size)
        return dict(
            affine_params=params,
            interpolation=self.torch_affine.interpolation
        )