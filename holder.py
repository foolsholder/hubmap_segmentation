import torch
import torchmetrics
import wandb
import pl_bolts

import pytorch_lightning as pl

from torch.nn import functional as F
from typing import (
    Dict, Optional, List, Tuple,
    Callable, Union, Any, Sequence
)
from metrics.dice_metric import Dice
from losses import BCELoss, SigmoidSoftDiceLoss, LossAggregation, LossMetric, BinaryFocalLoss

from models.utils import create_model


class ModelHolder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
            smooth: float = 1e-7,
            thr: float = 0.5
    ):
        super(ModelHolder, self).__init__()
        self.segmentor: torch.nn.Module = create_model(config['model_cfg'])

        metrics = [Dice(thr, smooth)]
        self.metrics_names = []
        for metric in metrics:
            self.__setattr__(metric.name, metric)
            self.metrics_names += [metric.name]

        losses: List[LossMetric] = [
            SigmoidSoftDiceLoss(loss_name='sigmoid_soft_dice'),
            BinaryFocalLoss(loss_name='binary_focal_loss'),
            LossAggregation(
                {
                    'binary_focal_loss': 1.0,
                    'sigmoid_soft_dice': 1.0
                },
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
        preds: Dict[str, torch.Tensor] = self.forward(input_x)

        stage: str = 'train' if self.segmentor.training else 'valid'

        for loss_name in self.losses_names:
            loss = self.__getattr__(loss_name)
            dct = loss.calc_loss_and_update_state(
                preds,
                batch_dict,
                stage=stage
            )
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
            self.log(metric.name + '/valid', metric.compute(), prog_bar=True)
            metric.reset()
        self.log_and_reset_losses()

    def log_and_reset_losses(self) -> None:
        for loss_name in self.losses_names:
            loss = self.__getattr__(loss_name)
            for stage in ['train', 'valid']:
                dct = loss.compute_loader_and_name(stage)
                for k, v in dct.items():
                    self.log(k, v, prog_bar=True)
            loss.reset()

    def forward(self, input_x: torch.Tensor) -> Dict[str, Any]:
        preds: Dict[str, torch.Tensor] = self.segmentor(input_x)
        preds['probs'] = torch.sigmoid(preds['logits'])
        return preds

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.segmentor.parameters(),
            lr=1e-3,
            weight_decay=1e-3
        )
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=20,
            warmup_start_lr=1e-6,
            max_epochs=-1
        )
        return [optim], [scheduler]
