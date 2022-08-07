import torch
import torch.nn as nn

from torch.nn import functional as F

from .modules import conv3x3_bn_relu
from typing import List, Union, Tuple
from ..tom import DecodeBlock, CenterBlock


class UPerDecoder(nn.Module):
    def __init__(self,
        in_dim: List[int], # len(in_dim) == 4
        decoder_out_channels: List[int],
        center_channels: int,
        last_channels: int
    ):
        super(UPerDecoder, self).__init__()
        # H // 4, W // 4 -> 512 // 4 = 128
        # H // 8, W // 8
        # H // 16, W // 16
        # H // 32, W // 32 -> 512 // 32 = 16
        #
        # 16 -> 32 -> 64 -> 128
        # f_1(16x16) - 32x32
        # cat 32x32
        # f_2(32x32) - 64x64
        # cat 64x64
        # f_3(64x64) - 128x128
        # cat 128x128
        # f_4(128x128) - 256x256
        # g(256x256) -> 512x512

        self.center = CenterBlock(
            in_channel=in_dim[-1],
            out_channel=center_channels
        )

        self.f_1 = DecodeBlock(
            in_channel=center_channels + in_dim[-1],
            out_channel=decoder_out_channels[0],
            upsample=True
        )
        self.f_2 = DecodeBlock(
            in_channel=decoder_out_channels[0] + in_dim[-2],
            out_channel=decoder_out_channels[1],
            upsample=True
        )
        self.f_3 = DecodeBlock(
            in_channel=decoder_out_channels[1] + in_dim[-3],
            out_channel=decoder_out_channels[2],
            upsample=True
        )
        self.f_4 = DecodeBlock(
            in_channel=decoder_out_channels[2] + in_dim[-4],
            out_channel=decoder_out_channels[3],
            upsample=True
        )
        self.g = DecodeBlock(
            in_channel=decoder_out_channels[3],
            out_channel=last_channels,
            upsample=True
        )

    def forward(
            self,
            feature: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        last_feat = feature[-1]
        # [batch_size; ; H // 32; H // 32]
        last_feat = self.center(last_feat)
        # [batch_size; ; H // 32; H // 32]
        cat_center = torch.cat([last_feat, feature[-1]], dim=1)

        out_f_1 = self.f_1(cat_center)
        # [batch_size; ; H // 16; H // 16]
        cat_f_1 = torch.cat([out_f_1, feature[-2]], dim=1)

        out_f_2 = self.f_2(cat_f_1)
        # [batch_size; ; H // 8; H // 8]
        cat_f_2 = torch.cat([out_f_2, feature[-3]], dim=1)

        out_f_3 = self.f_3(cat_f_2)
        # [batch_size; ; H // 4; H // 4]
        cat_f_3 = torch.cat([out_f_3, feature[-4]], dim=1)

        out_f_4 = self.f_4(cat_f_3)
        # [batch_size; decoder_out_channels[3]; H // 2; W // 2]

        last_feat_decoder = self.g(out_f_4)
        # [batch_size; last_channels; H; W]

        return (
            last_feat_decoder,
            [out_f_1, out_f_2, out_f_3, out_f_4]
        )
