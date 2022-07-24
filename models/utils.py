import torch

from typing import Dict, Any, Optional

from .tom import build_model as tom_build


def create_model(model_cfg: "Model_cfg"):
    possible_models = {
        'tom': tom_build
    }
    type = model_cfg.pop('type')
    return possible_models[type](**model_cfg)
