import os
import sys

import torch
from sys import argv

from hubmap_segmentation.models.swin.basic_modules import (
    Swin_V2_S_Weights, swin_v2_s
)
from hubmap_segmentation.models.swin.backbone import swin_s as our_swin


if __name__ == '__main__':
    if 'PRETRAINED' in os.environ:
        path_to_save = os.environ['PRETRAINED']
    else:
        path_to_save = argv[1]
    sys.path.append('..')
    torch_model = swin_v2_s(weights=Swin_V2_S_Weights.IMAGENET1K_V1)
    our_model = our_swin()

    our_model.input_conv.load_state_dict(torch_model.features[0].state_dict())

    for idx, layer_name in enumerate(our_model.layers_names):
        layer = our_model.__getattr__(layer_name)
        print(idx, flush=True)
        layer.load_state_dict(torch_model.features[2 * idx + 1].state_dict())
        if idx != 3:
            merge_layer = our_model.__getattr__(f'merge_{idx}')
            merge_layer.load_state_dict(torch_model.features[2 * idx + 2].state_dict())
    torch.save(
        our_model.state_dict(),
        os.path.join(path_to_save, 'swin_vs_small_imagenet.pth')
    )
