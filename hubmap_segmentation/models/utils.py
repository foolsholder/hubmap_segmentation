import torch

from typing import Dict, Any, Optional, Union
from .unet_segmentor import UNetSegmentor
from .segformer import SegFormer

def create_model(model_cfg: Dict[str, Union[Dict, Any]]):

    #return UNetSegmentor(**model_cfg)
    return SegFormer(
    config = model_cfg,
    in_channels=3,
    widths=[64, 128, 256, 512],
    depths=[3, 4, 6, 3],
    all_num_heads=[1, 2, 4, 8],
    patch_sizes=[7, 3, 3, 3],
    overlap_sizes=[4, 2, 2, 2],
    reduction_ratios=[8, 4, 2, 1],
    mlp_expansions=[4, 4, 4, 4],
    decoder_channels=256,
    scale_factors=[8, 4, 2, 1],
    num_classes=model_cfg['num_classes'],
)
