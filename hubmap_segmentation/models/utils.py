from typing import Dict, Any, Union

import torch

from hubmap_segmentation.models.unet.unet_segmentor import UNetSegmentor
from hubmap_segmentation.models.upernet.uper_net import UperNet


def create_model(model_cfg: Dict[str, Union[Dict, Any]]):
    if 'type' not in model_cfg:
        model_cfg['type'] = 'unet'

    type = model_cfg.pop('type')
    pretrained = ''
    if 'pretrained' in model_cfg:
        pretrained = model_cfg.pop('pretrained')

    if type == 'unet':
        model = UNetSegmentor(**model_cfg)
    else:
        model = UperNet(**model_cfg)

    if pretrained:
        import os
        model.load_state_dict(
            torch.load(
                os.path.join(
                    os.environ['PRETRAINED'],
                    pretrained
                ),
                map_location='cpu'
            ),
            strict=False
        )

    return model
