from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import PIL

from albumentations.core.transforms_interface import BasicTransform


NumType = Union[int, float, np.ndarray]
BoxType = Tuple[float, float, float, float]
KeypointType = Tuple[float, float, float, float]


class FDualTransform(BasicTransform):
    """Transform for segmentation task."""

    @property
    def targets(self) -> Dict[str, Callable]:
        return {
            "image": self.apply,
            "mask": self.apply_to_mask,
            "masks": self.apply_to_masks,
            "bboxes": self.apply_to_bboxes,
            "keypoints": self.apply_to_keypoints,
        }

    def apply_to_bbox(self, bbox: BoxType, **params) -> BoxType:
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint: KeypointType, **params) -> KeypointType:
        raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

    def apply_to_bboxes(self, bboxes: Sequence[BoxType], **params) -> List[BoxType]:
        return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]  # type: ignore

    def apply_to_keypoints(self, keypoints: Sequence[KeypointType], **params) -> List[KeypointType]:
        return [self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:]) for keypoint in keypoints]  # type: ignore # noqa

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        max_elem = np.max(img)
        if max_elem > 0:
            img = img / max_elem
        augmented_mask = self.apply(img, **{k: cv2.INTER_LINEAR if k == "interpolation" else v for k, v in params.items()})
        augmented_mask = (augmented_mask > 0).astype(img.dtype) * max_elem
        return augmented_mask

    def apply_to_masks(self, masks: Sequence[np.ndarray], **params) -> List[np.ndarray]:
        return [self.apply_to_mask(mask, **params) for mask in masks]