import torch

from typing import Dict, Any, Optional


def create_simple_network(
        model_dict: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
    model = torch.hub.load(
        repo_or_dir='mateuszbuda/brain-segmentation-pytorch',
        model='unet',
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=False
    )
    return model


def create_model(model_cfg: "Model_cfg"):
    from .tom import build_model
    return build_model(
        model_name='seresnext101',
        resolution=(512, 512),
        load_weights=True
    )
    """
    possible_models = {
        'simple': create_simple_network
    }
    type = model_cfg.pop('type')
    return possible_models[type](model_cfg)
    """
