from typing import Dict, Any, Union

import torch
import os

from hubmap_segmentation.models.unet.unet_segmentor import UNetSegmentor
from hubmap_segmentation.models.upernet.uper_net import UperNet


def replace_segmentor_to_load(ckpt):
    ckpt = ckpt['state_dict']
    res = type(ckpt)()
    patt = 'segmentor.'
    pref = len(patt)
    for k, v in ckpt.items():
        res[k[pref:]] = v
    return res

def create_model(model_cfg: Dict[str, Union[Dict, Any]]):
    if 'type' not in model_cfg:
        model_cfg['type'] = 'unet'

    type = model_cfg.pop('type')
    pretrained = ''
    if 'pretrained' in model_cfg:
        pretrained = model_cfg.pop('pretrained')
        if 'from_freeze' in pretrained:
            pretrained = pretrained.replace('from_freeze', os.environ['SHUBMAP_EXPS'])
            freeze = True
        else:
            pretrained = os.path.join(os.environ['PRETRAINED'], pretrained)
            freeze = False
        ckpt = torch.load(
            pretrained,
            map_location='cpu'
        )
        if freeze:
            ckpt = replace_segmentor_to_load(ckpt)

    if type == 'unet':
        model = UNetSegmentor(**model_cfg)
    else:
        model = UperNet(**model_cfg)

    if pretrained:
        model.load_state_dict(
            ckpt,
            strict=False
        )

    return model
