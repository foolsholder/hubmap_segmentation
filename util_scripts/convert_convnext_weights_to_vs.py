import os
import sys

import torch
from sys import argv

from torchvision.models.convnext import (
    ConvNeXt_Base_Weights, convnext_base
)
from hubmap_segmentation.models.convnext.backbone import convnext_base as our_convnext


if __name__ == '__main__':
    if 'PRETRAINED' in os.environ:
        path_to_save = os.environ['PRETRAINED']
    else:
        path_to_save = argv[1]
    sys.path.append('..')
    torch_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    our_model = our_convnext()

    our_model.input_conv.load_state_dict(torch_model.features[0].state_dict())

    for idx, layer_name in enumerate(our_model.layers_names):
        layer = our_model.__getattr__(layer_name)
        print(idx, flush=True)
        layer.load_state_dict(torch_model.features[2 * idx + 1].state_dict())
        if idx != 3:
            stride_layer = our_model.__getattr__(f'stride_{idx + 1}')
            stride_layer.load_state_dict(torch_model.features[2 * idx + 2].state_dict())
    torch.save(
        our_model.state_dict(),
        os.path.join(path_to_save, 'convnext_vs_base_imagenet.pth')
    )
