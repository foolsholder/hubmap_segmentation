import torch
import torch.nn as nn

from torch.nn import functional as F

from typing import List, Union, Tuple
from .decoder_modules import \
    DecodeBlock, CenterBlock, \
    FFCDecodeBlock, FFCCenterBlock


class UNetDecoder(nn.Module):
    def __init__(self,
        in_dim: List[int], # len(in_dim) == 4
        decoder_out_channels: List[int],
        upsamples: List[bool],
        center_channels: int,
        last_channels: int
    ):
        super(UNetDecoder, self).__init__()
        self.center = FFCCenterBlock(
            in_channel=in_dim[-1],
            out_channel=center_channels
        )

        self.layers_names = []
        prev = center_channels
        for idx in range(len(decoder_out_channels)):
            layer = FFCDecodeBlock(
                in_channel=prev + in_dim[-1-idx],
                out_channel=decoder_out_channels[idx],
                upsample=upsamples[idx]
            )
            prev = decoder_out_channels[idx]
            layer_name = 'f_{}'.format(idx + 1)
            self.__setattr__(layer_name, layer)
            self.layers_names += [layer_name]
        self.g = FFCDecodeBlock(
            in_channel=prev,
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
        last_feat = cat_center

        outs = []

        for idx, layer_name in enumerate(self.layers_names):
            idx = idx + 1 # f_1, f_2 ...
            out = getattr(self, layer_name)(last_feat)
            outs += [out]
            if idx + 1 <= len(feature):
                last_feat = torch.cat([out, feature[-1-idx]], dim=1)
            else:
                last_feat = out

        #out_f_1 = self.f_1(cat_center)
        # [batch_size; ; H // 16; H // 16]
        #cat_f_1 = torch.cat([out_f_1, feature[-2]], dim=1) - idx 0

        #out_f_2 = self.f_2(cat_f_1)
        # [batch_size; ; H // 8; H // 8]
        #cat_f_2 = torch.cat([out_f_2, feature[-3]], dim=1) - idx 1

        #out_f_3 = self.f_3(cat_f_2)
        # [batch_size; ; H // 4; H // 4]
        #cat_f_3 = torch.cat([out_f_3, feature[-4]], dim=1) - idx 2

        #out_f_4 = self.f_4(cat_f_3) - idx 3
        # [batch_size; decoder_out_channels[3]; H // 2; W // 2]

        #last_feat_decoder = self.g(out_f_4)
        # [batch_size; last_channels; H; W]
        #print(last_feat.shape, flush=True)
        last_feat_decoder = self.g(last_feat)

        return (
            last_feat_decoder,
            outs
        )
