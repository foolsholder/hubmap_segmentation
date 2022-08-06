import torch

from typing import Dict, Any, Optional

from .tom import build_model as tom_build
from .effnet import create_effnet_segmentor
from .swin_uper import create_swin_upernet


def create_model(model_cfg: "Model_cfg"):
    possible_models = {
        'tom': tom_build,
        'effnet': create_effnet_segmentor,
        'swin': create_swin_upernet
    }
    type = model_cfg.pop('type')
    return possible_models[type](**model_cfg)
