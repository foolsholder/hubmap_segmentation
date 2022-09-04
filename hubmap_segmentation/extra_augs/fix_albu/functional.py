import cv2
from PIL import Image
from PIL.Image import AFFINE
import PIL
import numpy as np
from torchvision.transforms import RandomAffine
import torchvision.transforms.functional as F


def resize(img, height, width, interpolation=PIL.Image.Resampling.BILINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    img = np.clip(img, 0, 255)
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((width, height), resample=interpolation)
    return np.array(img)


class TorchAffine(RandomAffine):
    def right_forward(self, img, interpolation, params):
        return F.affine(img, *params, interpolation=interpolation, fill=self.fill, center=self.center)