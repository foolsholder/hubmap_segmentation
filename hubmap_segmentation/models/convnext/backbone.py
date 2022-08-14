import torch
import math
import warnings
import copy

import os
from torch import nn


from torchvision.models.convnext import (
    CNBlockConfig, _log_api_usage_once, LayerNorm2d,
    Conv2dNormActivation, partial, Tensor, WeightsEnum,
    _ovewrite_named_param, ConvNeXt_Small_Weights
)

from .basic_modules import VSCNBlock


from typing import Dict, Any, Optional, Tuple, List, Sequence, Union, Callable


class ConvNeXtVS(nn.Module):
    def __init__(
        self,
        block_setting: List[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        block: Optional[Callable[..., nn.Module]] = VSCNBlock,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = VSCNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        self.input_conv = Conv2dNormActivation(
            3,
            firstconv_output_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=norm_layer,
            activation_layer=None,
            bias=True,
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        self.layers_names = []

        for idx, cnf in enumerate(block_setting):
            # Bottlenecks
            substage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                substage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            stage_layer = nn.Sequential(*substage)
            if cnf.out_channels is not None:
                # Downsampling
                stride_layer = nn.Sequential(
                    norm_layer(cnf.input_channels),
                    nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                )
                stride_name = f'stride_{idx + 1}'
                self.__setattr__(stride_name, stride_layer)
            layer_name = f'layer_{idx + 1}'
            self.__setattr__(layer_name, stage_layer)
            self.layers_names += [layer_name]

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.input_conv(x).contiguous()
        res = []
        for idx, layer_name in enumerate(self.layers_names):
            layer = self.__getattr__(layer_name)
            x = layer(x)#.contiguous()
            res += [x]
            if idx + 1 != len(self.layers_names):
                x = self.__getattr__(f'stride_{idx + 1}')(x).contiguous()
        return res


def _convnext_vs(
    block_setting: List[CNBlockConfig],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ConvNeXtVS:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ConvNeXtVS(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def convnext_vs_small(
    *, weights: Optional[ConvNeXt_Small_Weights] = None, progress: bool = True, **kwargs: Any
    ) -> ConvNeXtVS:
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.

    Args:
        weights (:class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.convnext.ConvNeXt_Small_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.convnext.ConvNext``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ConvNeXt_Small_Weights
        :members:
    """
    weights = ConvNeXt_Small_Weights.verify(weights)

    block_setting = [
        # [H, W] -> [H // 4; W // 4]
        CNBlockConfig(96, 192, 3), # [H // 4; W // 4] -> [H // 8; W // 8]
        CNBlockConfig(192, 384, 3), # [H // 8; W // 8] -> [H // 16; W // 16]
        CNBlockConfig(384, 768, 27), # [H // 16; W // 16] -> [H // 32; W // 32]
        CNBlockConfig(768, None, 3), # [H // 32; W // 32] -> [H // 32; W // 32]
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext_vs(block_setting, stochastic_depth_prob, weights, progress, **kwargs)


def create_convnext(load_weights: str = '') -> ConvNeXtVS:
    model = convnext_vs_small()
    if load_weights == 'imagenet':
        import os
        model.load_state_dict(
            torch.load(
                os.path.join(os.environ['PRETRAINED'],
                'convnext_vs_small_imagenet.pth'),
                map_location='cpu'
            )
        )
    return model
