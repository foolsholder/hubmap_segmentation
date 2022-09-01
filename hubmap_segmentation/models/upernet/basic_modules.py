# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 inplace: bool = True
                 ):
        super().__init__()
        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(
            self,
            pool_scales,
            in_channels,
            channels
    ):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                    )))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
