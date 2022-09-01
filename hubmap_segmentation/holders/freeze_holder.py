import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from copy import copy
from typing import Dict, List, Union, Tuple, Optional, Any, Callable

from .holder import ModelHolder
from .optimizer_utils import create_opt_shed


class FreezeHolder(ModelHolder):
    def __init__(
            self,
            freeze_params: 'str' = 'only_bb', # 'only_bb' or 'wo_conv_seg'
            model_type: 'str' = 'unet',
            **kwargs
    ):
        super(FreezeHolder, self).__init__(**kwargs)
        self._freeze_params = freeze_params
        self._model_type = model_type

    def configure_optimizers(self):
        assert self._freeze_params == 'only_bb'
        if self._model_type == 'upernet':
            params = [
                {
                    'params': self.segmentor.auxiliary_head.parameters(),
                },
                {
                    'params': self.segmentor.decode_head.parameters(),
                },
            ]
        else:
            params = [
                {
                    'params': self.segmentor.decoder.parameters(),
                },
                {
                    'params': self.segmentor.final_conv.parameters(),
                },
                {
                    'params': self.segmentor.aux_head.parameters(),
                },
            ]
        return create_opt_shed(self._config['opt_sched'], params)

    def freeze_layers(self):
        if self._freeze_params == 'only_bb':
            self.segmentor.backbone.eval()

    def on_train_start(self) -> None:
        def turn_off_grads(module):
            for name, param in module.named_parameters():
                param.requires_grad = False
        if self._freeze_params == 'only_bb':
            turn_off_grads(self.segmentor.backbone)
        if self._freeze_params == 'wo_conv_seg':
            raise NotImplemented

    def on_train_epoch_start(self) -> None:
        super(FreezeHolder, self).on_train_epoch_start()
        self.freeze_layers()
