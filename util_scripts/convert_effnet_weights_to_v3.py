import os
import sys

import torch
from sys import argv

from torchvision.models.efficientnet import (
    efficientnet_v2_m, efficientnet_v2_s,
    EfficientNet_V2_M_Weights, EfficientNet_V2_S_Weights
)
from hubmap_segmentation.models.effnet.backbone import efficientnet_v2_m as our_effnet


if __name__ == '__main__':
    if 'PRETRAINED' in os.environ:
        path_to_save = os.environ['PRETRAINED']
    else:
        path_to_save = argv[1]
    sys.path.append('..')
    torch_model = efficientnet_v2_m(EfficientNet_V2_M_Weights.IMAGENET1K_V1)
    our_model = our_effnet()

    our_model.input_conv.load_state_dict(torch_model.features[0].state_dict())

    for idx, layer_name in enumerate(our_model.layers_names):
        our_model.__getattr__(layer_name).load_state_dict(torch_model.features[idx + 1].state_dict())
    torch.save(
        our_model.state_dict(),
        os.path.join(path_to_save, 'effnet_v3_imagenet.pth')
    )
