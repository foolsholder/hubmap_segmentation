import torch
import torchmetrics
import wandb
import pl_bolts

import numpy as np
import pytorch_lightning as pl

from copy import deepcopy, copy
from torch.nn import functional as F
from typing import (
    Dict, Optional, List, Tuple,
    Callable, Union, Any, Sequence
)
from hubmap_segmentation.metrics.dice_metric import Dice
from hubmap_segmentation.losses import (
    BCELoss, SigmoidSoftDiceLoss, NLSDLoss,
    LossAggregation, CATCELoss, CatSoftDiceLoss,
    BinaryFocalLoss, TverskyLoss, CatFocalLoss,
    LovaszHingeLoss, SymmetricUnifiedFocalLoss
)
from hubmap_segmentation.holders.holder import ModelHolder
from hubmap_segmentation.models.utils import create_model

available_losses = {
    'bce': BCELoss,
    'binary_focal_loss': BinaryFocalLoss,
    'sigmoid_soft_dice': SigmoidSoftDiceLoss,
    'tversky_loss': TverskyLoss,
    'unified_focal_loss': SymmetricUnifiedFocalLoss,
    'cat_lovasz': LovaszHingeLoss,
    'cat_soft_dice': CatSoftDiceLoss,
    'cat_focal_loss': CatFocalLoss,
    'catce': CATCELoss,
    'negative_log_cat_soft_dice': NLSDLoss
}


class DistillHolder(ModelHolder):
    def __init__(
            self,
            config: Dict[str, Any],
            **kwargs
    ):
        teacher_conf = deepcopy(config['teacher_cfg'])
        super(DistillHolder, self).__init__(config, **kwargs)

        self.teacher = create_model(teacher_conf)
        for name, param in self.teacher.parameters():
            param.requires_grad = False

    def on_train_epoch_start(self) -> None:
        self.teacher.eval()

    def _step_logic(
            self,
            batch_dict: Dict[str, torch.Tensor],
            stage: str
    ) -> Dict[str, Any]:
        input_x = batch_dict['input_x']

        preds: Dict[str, torch.Tensor] = self.forward(input_x, additional_info=batch_dict, stage=stage)

        if stage != 'valid':
            with torch.no_grad():
                # only_train
                teacher_preds = self.teacher(input_x, additional_info=batch_dict)
                for k, v in teacher_preds.items():
                    preds[k + '_teacher'] = v

            self._log_losses(batch_dict, preds, stage)
        return preds
