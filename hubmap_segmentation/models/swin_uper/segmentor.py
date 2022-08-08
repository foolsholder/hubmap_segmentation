import torch
from torch import nn
from torch.nn import functional as F

from .backbone import SwinTransformerV1, LayerNorm2d
from .decoder import UPerDecoder


class SwinUperNet(nn.Module):
    def __init__(
        self,
        size: str = 'tiny',
        use_aux_head: bool = False,
        num_classes: int = 1
    ):
        super(SwinUperNet, self).__init__()
        depths = [2, 2, 6, 2]
        if size == 'small':
            depths = [2, 2, 18, 2]
        self.encoder = SwinTransformerV1(depths=depths)

        encoder_dim = [96, 192, 384, 768]
        decoder_out = [256, 256, 128, 128]
        center_channels = 256
        last_channels = 64
        self.decoder = UPerDecoder(
            in_dim=encoder_dim,
            decoder_out_channels=decoder_out,
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


def create_swin_upernet(
        load_weights: str = '',
        size: str = 'tiny',
        use_aux_head: bool = False,
        num_classes: int = 1
    ):
    model = SwinUperNet(
        size=size,
        use_aux_head=use_aux_head,
        num_classes=num_classes
    )
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
