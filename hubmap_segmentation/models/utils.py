from typing import Dict, Any, Union
from hubmap_segmentation.models.unet.unet_segmentor import UNetSegmentor


def create_model(model_cfg: Dict[str, Union[Dict, Any]]):
    return UNetSegmentor(**model_cfg)
