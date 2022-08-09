import torch
from torch import nn
from torch.nn import functional as F


from typing import Dict, Any, List, Optional, Tuple, Sequence, Union
from .swin_v1.backbone import create_swin_v1
from .effnet.backbone import create_effnet
from .unet_decoder import UNetDecoder


avaliable_backbones = {
    'swin': create_swin_v1,
    'effnet': create_effnet
}

backboned_unet_args = {
    'swin': {
        'encoder_out' : [96, 192, 384, 768],
        'decoder_out' : [256, 256, 128, 128],
        'upsamples' : [True, True, True, True][::-1],
        'center_channels' : 256,
        'last_channels' : 64
    },
    'effnet': {
        'encoder_out' : [24, 48, 80, 160, 176, 304, 512],
        'decoder_out' : [512, 320, 176, 160, 120, 90, 64],
        'upsamples' : [False, True, True, True, False, True, False][::-1],
        'center_channels' : 256,
        'last_channels' : 64
    }
}


def create_backbone(backbone_cfg: Dict[str, Any]):
    type_bb = backbone_cfg.pop('type')
    return avaliable_backbones[type_bb](**backbone_cfg), \
            backboned_unet_args[type_bb]


class UNetSegmentor(nn.Module):
    def __init__(
        self,
        backbone_cfg: Dict[str, Any],
        use_aux_head: bool = False,
        num_classes: int = 1
    ):
        super(UNetSegmentor, self).__init__()
        self.encoder, unet_args = create_backbone(backbone_cfg)

        encoder_out = unet_args['encoder_out']
        decoder_out = unet_args['decoder_out']
        upsamples = unet_args['upsamples']
        center_channels = unet_args['center_channels']
        last_channels = unet_args['last_channels']

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
            aux_logits = F.interpolate(
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
