import torch
from torch import nn
from torch.nn import functional as F

from .backbone import SwinTransformerV1, LayerNorm2d
from .decoder import UPerDecoder


class SwinUperNet(nn.Module):
    def __init__(self, size: str = 'tiny'):
        super(SwinUperNet, self).__init__()
        depths = [2, 2, 6, 2]
        if size == 'small':
            depths = [2, 2, 18, 2]
        self.encoder = SwinTransformerV1(depths=depths)
        encoder_dim = [96, 192, 384, 768]

        self.decoder = UPerDecoder(
            in_dim=encoder_dim,
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=256
        )

        self.logit = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, input_x: torch.Tensor):
        x = input_x
        encoder_feats = self.encoder(x)
        last, decoder_feats = self.decoder(encoder_feats)

        logit = self.logit(last)

        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)

        return {
            "logits": logit,
            "probs": torch.sigmoid(logit)
        }


def create_swin_upernet(load_weights: str = '', size: str = 'tiny'):
    model = SwinUperNet(size=size)
    if load_weights == 'frog':
        import os
        model.encoder.load_state_dict(
            torch.load(
                os.path.join(
                    os.environ['PRETRAINED'],
                    'swin_{}_patch4_window7_224_22k.pth'.format(size)
                ),
                map_location='cpu'
            )['model'],
            strict=False
        )
    return model
