import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from typing import Dict, List, Sequence, Optional, Any

from .backbone import efficientnet_v2_m
from ..unet_decoder import UNetDecoder


class EfficientNetSegmentor(nn.Module):
    def __init__(
            self,
            load_weights: str = '',
            use_aux_head: bool = False,
            num_classes: int = 1
    ):
        super(EfficientNetSegmentor, self).__init__()
        self.encoder = efficientnet_v2_m()
        if load_weights == 'imagenet':
            self.encoder.load_state_dict(
                torch.load(
                    os.path.join(
                        os.environ['PRETRAINED'],
                        'effnet_v3_imagenet.pth'),
                    map_location='cpu'))
        # -> 3, 512
        # inp_conv -> 24, 256
        # block0 -> 24, 256   |  f_7 -> 256
        # block1 -> 48, 128   |  f_6 -> 256
        # block2 -> 80, 64    |  f_5 -> 128
        # block3 -> 160, 32   |  f_4 -> 64
        # block4 -> 176, 32   |  f_3 -> 32
        # block5 -> 304, 16   |  f_2 -> 32
        # block6 -> 512, 16   |  f_1 -> 16

        encoder_out = [24, 48, 80, 160, 176, 304, 512]
        decoder_out = [512, 320, 176, 160, 120, 90, 64]
        upsamples = [False, True, True, True, False, True, False][::-1]
        center_channels = 256
        last_channels = 64

        self.decoder = UNetDecoder(
            in_dim=encoder_out,
                    #  C +  C +   C + C  + C  + C +  C <-
            decoder_out_channels=decoder_out,
            #scale_factors=[2, 2, 2, 2, 1, 2, 1],
            upsamples=upsamples,
            center_channels=center_channels,
            last_channels=last_channels
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                last_channels,
                last_channels,
                kernel_size=3,
                padding=1,
                #bias=False
            ),
            nn.BatchNorm2d(last_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_channels, num_classes, kernel_size=1)
        )
        self.use_aux_head = use_aux_head
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Conv2d(
                    decoder_out[-1],
                    last_channels,
                    kernel_size=3,
                    padding=1,
                    #bias=False
                ),
                nn.BatchNorm2d(last_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_channels, num_classes, kernel_size=1)
            )

    def forward(self, input_x: torch.Tensor):
        x = input_x
        encoder_feats = self.encoder(x)
        last, decoder_feats = self.decoder(encoder_feats)

        logits = self.final_conv(last)

        res = {
            "logits": logits,
            "probs": torch.sigmoid(logits)
        }
        if self.use_aux_head:
            aux_logits = self.aux_head(decoder_feats[-1])
            aux_logits = F.upsample(
                aux_logits,
                size=logits.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            res.update({
                "aux_logits": aux_logits,
                "aux_probs": torch.sigmoid(aux_logits)
            })
        return res


def create_effnet_segmentor(
        load_weights: str = 'imagenet',
        use_aux_head: bool = False,
        num_classes: int = 1
    ) -> nn.Module:
    return EfficientNetSegmentor(
        load_weights=load_weights,
        use_aux_head=use_aux_head,
        num_classes=num_classes
    )
