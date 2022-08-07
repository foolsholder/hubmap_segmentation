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
    BCELoss, SigmoidSoftDiceLoss,
    LossAggregation, LossMetric,
    BinaryFocalLoss, TverskyLoss,
    LovaszHingeLoss, SymmetricUnifiedFocalLoss
)
from hubmap_segmentation.holders.optimizer_utils import create_opt_shed
from hubmap_segmentation.models.utils import create_model

available_losses = {
    'bce': BCELoss,
    'binary_focal_loss': BinaryFocalLoss,
    'sigmoid_soft_dice': SigmoidSoftDiceLoss,
    'tversky_loss': TverskyLoss,
    'unified_focal_loss': SymmetricUnifiedFocalLoss,
    'lovasz_hinge_loss': LovaszHingeLoss
}


class ModelHolder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
            smooth: float = 1e-7,
            tiling_height: int = 512,
            tiling_width: int = 512,
            thr: float = 0.5
    ):
        super(ModelHolder, self).__init__()
        self._config = deepcopy(config)
        self.segmentor: torch.nn.Module = create_model(config['model_cfg'])
        self.tiling_height = tiling_height
        self.tiling_width = tiling_width
        metrics = [Dice(thr, smooth)]
        self.metrics_names = []
        for metric in metrics:
            self.__setattr__(metric._name, metric)
            self.metrics_names += [metric._name]

        self._stages_names = ['train', 'valid']

        losses: List[LossMetric] = []
        if 'losses' in config:
            losses = [
                *[available_losses[loss]() for loss in config['losses']['names']],
            ]

        aux_losses: List[LossMetric] = []
        if 'aux' in config['losses'] \
                and config['losses']['aux'] > 0. \
                and len(losses) > 0:
            for loss in losses:
                aux_loss = type(loss)(
                    loss_name='aux_' + loss._name
                )
                aux_losses += [aux_loss]
            aux_weight = config['losses'].pop('aux')

            loss_names = copy(config['losses']['names'])
            loss_weigths = copy(config['losses']['weights'])

            for loss_name, loss_weigth in zip(loss_names, loss_weigths):
                config['losses']['names'] += ['aux_' + loss_name]
                config['losses']['weights'] += [loss_weigth * aux_weight]

        losses += aux_losses
        if len(losses) > 0:
            losses += [
                LossAggregation(
                    dict(zip(config['losses']['names'], config['losses']['weights'])),
                    loss_name='loss'
                )
            ]


        self.losses_names = []
        for loss in losses:
            self.__setattr__(loss._name, loss)
            self.losses_names += [loss._name]

    def training_step(
            self,
            batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        return self._step_logic(batch_dict)

    def _step_logic(
            self,
            batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        input_x = batch_dict['input_x']

        stage: str = 'train' if self.segmentor.training else 'valid'
        self._stage = stage
        preds: Dict[str, torch.Tensor] = self.forward(input_x, additional_info=batch_dict)

        for loss_name in self.losses_names:
            loss = self.__getattr__(loss_name)
            dct = loss.calc_loss_and_update_state(
                preds,
                batch_dict,
                stage=stage
            )
            if stage != 'valid':
                for k, v in dct.items():
                    self.log('{}_batch/{}'.format(k, stage), v, prog_bar=True)
            preds.update(dct)

        return preds

    def validation_step(
            self,
            batch_dict: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, Any]:
        preds = self._step_logic(batch_dict)
        for metric_name in self.metrics_names:
            metric = self.__getattr__(metric_name)
            metric.update(preds, batch_dict)
        return preds

    def validation_epoch_end(
            self,
            outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ) -> None:
        for metric_name in self.metrics_names:
            metric = self.__getattr__(metric_name)
            res_metric = metric.compute_every()
            for k, v in res_metric.items():
                self.log(k, v, prog_bar=True)
            metric.reset()
        self.log_and_reset_losses()

    def log_and_reset_losses(self) -> None:
        for loss_name in self.losses_names:
            loss = self.__getattr__(loss_name)
            for stage in self._stages_names:
                dct = loss.compute_loader_and_name(stage)
                for k, v in dct.items():
                    self.log(k, v, prog_bar=True)
            loss.reset()

    def forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        preds: Dict[str, torch.Tensor] = self.segmentor(input_x)
        return preds

    def configure_optimizers(self):
        return create_opt_shed(self._config['opt_sched'], self.segmentor.parameters())


