import torch
import math
import warnings
import copy

import os
from torch import nn

from torchvision.models.regnet import (
    _log_api_usage_once, _ovewrite_named_param, BlockParams,
    SimpleStemIN, ResBottleneckBlock, AnyStage, OrderedDict,
    Tensor, WeightsEnum, partial, RegNet_Y_8GF_Weights
)

from typing import Dict, Any, Optional, Tuple, List, Sequence, Union, Callable


class RegNetVS(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.input_conv = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        self.layers_names = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            layer_name = f"block{i+1}"
            layer = AnyStage(
                current_width,
                width_out,
                stride,
                depth,
                block_type,
                norm_layer,
                activation,
                group_width,
                bottleneck_multiplier,
                block_params.se_ratio,
                stage_index=i + 1,
            )
            self.__setattr__(layer_name, layer)
            self.layers_names += [layer_name]
            current_width = width_out

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.input_conv(x)
        res = []
        for layer_name in self.layers_names:
            layer = self.__getattr__(layer_name)
            x = layer(x)
            res += [x]
        return res


def _regnet_vs(
    block_params: BlockParams,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> RegNetVS:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    norm_layer = kwargs.pop("norm_layer", partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1))
    model = RegNetVS(block_params, norm_layer=norm_layer, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def regnet_y_8gf_vs(
        *,
        weights: Optional[RegNet_Y_8GF_Weights] = None,
        progress: bool = True,
        **kwargs: Any
    ) -> RegNetVS:
    """
    Constructs a RegNetY_8GF architecture from
    `Designing Network Design Spaces <https://arxiv.org/abs/2003.13678>`_.

    Args:
        weights (:class:`~torchvision.models.RegNet_Y_8GF_Weights`, optional): The pretrained weights to use.
            See :class:`~torchvision.models.RegNet_Y_8GF_Weights` below for more details and possible values.
            By default, no pretrained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to either ``torchvision.models.regnet.RegNet`` or
            ``torchvision.models.regnet.BlockParams`` class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py>`_
            for more detail about the classes.

    .. autoclass:: torchvision.models.RegNet_Y_8GF_Weights
        :members:
    """
    weights = RegNet_Y_8GF_Weights.verify(weights)

    params = BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25, **kwargs
    )
    return _regnet_vs(params, weights, progress, **kwargs)


def create_regnet_y(load_weights: str = '') -> RegNetVS:
    model = regnet_y_8gf_vs()
    if load_weights == 'imagenet':
        import os
        model.load_state_dict(
            torch.load(
                os.path.join(os.environ['PRETRAINED'],
                'regnet_vs_y_imagenet.pth'),
                map_location='cpu'
            )
        )
    return model
