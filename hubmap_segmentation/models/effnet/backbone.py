import torch
import math
import warnings
import copy

import os
from torch import nn

from torchvision.models.efficientnet import (
    _log_api_usage_once, _MBConvConfig, Conv2dNormActivation,
    EfficientNet, EfficientNet_V2_M_Weights, _ovewrite_named_param,
    partial, MBConvConfig, FusedMBConvConfig, WeightsEnum
)

from .basic_modules import _efficientnet_conf_v3, MBConvV3

from typing import Dict, Any, Optional, Tuple, List, Sequence, Union, Callable


class EfficientNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        stochastic_depth_prob: float = 0.2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet V1 and V2 main class
        Args:
            inverted_residual_setting (Sequence[Union[MBConvConfig, FusedMBConvConfig]]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        super().__init__()
        _log_api_usage_once(self)

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if "block" in kwargs:
            warnings.warn(
                "The parameter 'block' is deprecated since 0.13 and will be removed 0.15. "
                "Please pass this information on 'MBConvConfig.block' instead."
            )
            if kwargs["block"] is not None:
                for s in inverted_residual_setting:
                    if isinstance(s, MBConvConfig):
                        s.block = kwargs["block"]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.input_conv = Conv2dNormActivation(
            3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer, dilation=dilation))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        layers_names = []
        for idx, layer in enumerate(layers):
            layer_name = 'block_{}'.format(idx)
            self.__setattr__(layer_name, layer)
            layers_names += [layer_name]
        self.layers_names = layers_names

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        #res: Dict[str,torch.Tensor] = {}
        x = self.input_conv(x)
        res = []
        for layer_name in self.layers_names:
            layer = self.__getattr__(layer_name)
            x = layer(x)
            #res[layer_name] = x
            res += [x]
        return res


def _efficientnet(
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        last_channel: Optional[int],
        weights: Optional[WeightsEnum],
        progress: bool,
        dilation: int = 1,
        **kwargs: Any,
    ) -> EfficientNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = EfficientNetV3(inverted_residual_setting,
                           dropout,
                           last_channel=last_channel,
                           dilation=dilation,
                           **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def efficientnet_v2_m(
        *,
        weights: Optional[EfficientNet_V2_M_Weights] = None,
        progress: bool = True,
        dilation: int = 1,
        **kwargs: Any
    ) -> EfficientNetV3:
    """
    Constructs an EfficientNetV3-M architecture from
    `EfficientNetV3: Smaller Models and Faster Training <https://arxiv.org/abs/2104.00298>`_.
    """
    weights = EfficientNet_V2_M_Weights.verify(weights)

    inverted_residual_setting, last_channel = _efficientnet_conf_v3("efficientnet_v2_m")
    return _efficientnet(
        inverted_residual_setting,
        0.3,
        last_channel,
        weights,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
        dilation=dilation,
        **kwargs,
    )


def create_effnet(
        load_weights: str = '',
        dilation: int = 1
    ):
    model = efficientnet_v2_m(dilation=dilation)
    if load_weights == 'imagenet':
        model.load_state_dict(
            torch.load(
                os.path.join(
                    os.environ['PRETRAINED'],
                    'effnet_v3_imagenet.pth'),
                map_location='cpu'),
            strict=False
        )
    return model
