import os
import sys

import torch
from sys import argv

from torchvision.models.regnet import (
    RegNet_Y_8GF_Weights, regnet_y_8gf
)
from hubmap_segmentation.models.regnet.backbone import regnet_y_8gf_vs as out_regnet


if __name__ == '__main__':
    if 'PRETRAINED' in os.environ:
        path_to_save = os.environ['PRETRAINED']
    else:
        path_to_save = argv[1]
    sys.path.append('..')
    torch_model = regnet_y_8gf(weights=RegNet_Y_8GF_Weights.IMAGENET1K_V2)
    our_model = out_regnet()

    our_model.input_conv.load_state_dict(torch_model.stem.state_dict())

    for idx, layer_name in enumerate(our_model.layers_names):
        layer = our_model.__getattr__(layer_name)
        layer.load_state_dict(
            torch_model.trunk_output[idx].state_dict()
        )
    torch.save(
        our_model.state_dict(),
        os.path.join(path_to_save, 'regnet_vs_y_imagenet.pth')
    )
