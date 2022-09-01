# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


from typing import Dict, List

from .basic_modules import ConvModule


class FCNHead(nn.Module):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
            self,
            num_convs=1,
            kernel_size=3,
            dilation=1,
            num_classes=6,
            dropout_ratio=0.1,
            in_channels=384,
            in_index=2,
            channels=256
        ):
        super(FCNHead, self).__init__()
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs: List = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation)
        ]
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _forward_feature(self, inputs):
        feats = self.convs(inputs[self.in_index])
        if self.dropout:
            feats = self.dropout(feats)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        aux_logits = self.conv_seg(output)
        return aux_logits

