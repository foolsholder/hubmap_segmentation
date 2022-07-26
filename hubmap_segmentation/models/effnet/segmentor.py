import torch
import torch.nn as nn
import os

from typing import Dict, List, Sequence, Optional, Any

from .backbone import efficientnet_v2_m
from .decoder import MAnetDecoder


class EfficientNetSegmentor(nn.Module):
    def __init__(
            self,
            load_weights: str = ''
    ):
        super(EfficientNetSegmentor, self).__init__()
        self.backbone = efficientnet_v2_m()
        if load_weights == 'imagenet':
            self.backbone.load_state_dict(
                torch.load(
                    os.path.join(
                        os.environ['PRETRAINED'],
                        'effnet_v3_imagenet.pth'),
                    map_location='cpu'))
        # -> 3, 512
        # inp_conv -> 24, 256
        # block0 -> 24, 256
        # block1 -> 48, 128
        # block2 -> 80, 64
        # block3 -> 160, 32
        # block4 -> 176, 32
        # block5 -> 304, 16
        # block6 -> 512, 16
        self.decoder = MAnetDecoder(
            encoder_channels=[24, 48, 80, 160, 176, 304, 512],
                           #  C +  C +   C + C  + C  + C +  C <-
            decoder_channels=[512, 320, 176, 160, 80, 40, 40],
            scale_factors=[2, 2, 2, 2, 1, 2, 1],
            n_blocks=7,
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ELU(True),
            nn.Conv2d(24, 1, kernel_size=1),
        )

    def forward(self, input_x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_feats = self.backbone(input_x)
        feats = []
        for layer_name in self.backbone.layers_names:
            feats += [backbone_feats[layer_name]]
        decoder_out = self.decoder(*feats)
        logits = self.final_conv(decoder_out)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits)
        }


def create_effnet_segmentor(
        load_weights: str = 'imagenet'
    ) -> nn.Module:
    return EfficientNetSegmentor(load_weights=load_weights)
