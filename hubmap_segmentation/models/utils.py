import torch

from typing import Dict, Any, Optional, Union
from .unet_segmentor import UNetSegmentor


def create_model(model_cfg: Dict[str, Union[Dict, Any]]):
    return UNetSegmentor(**model_cfg)
