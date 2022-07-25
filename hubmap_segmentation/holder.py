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
from .metrics.dice_metric import Dice
from .losses import BCELoss, SigmoidSoftDiceLoss, LossAggregation, LossMetric, BinaryFocalLoss

from .models.utils import create_model


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
            self.__setattr__(metric._name, metric)
            self.metrics_names += [metric._name]

        self._stages_names = ['train', 'valid']

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

    def forward(self, input_x: torch.Tensor) -> Dict[str, Any]:
        preds: Dict[str, torch.Tensor] = self.segmentor(input_x)
        # preds['probs'] = torch.sigmoid(preds['logits'])
        return preds

    def configure_optimizers(self):
        optim = torch.optim.RAdam(
            self.segmentor.parameters(),
            lr=3e-4,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=20,
            warmup_start_lr=1e-6,
            max_epochs=-1
        )
        return [optim], [scheduler]


class TTAHolder(ModelHolder):
    def __init__(self, *args, **kwargs):
        super(TTAHolder, self).__init__(*args, **kwargs)
        self._stages_names = ['valid']

    def forward(self, input_x: torch.Tensor) -> Dict[str, Any]:
        """
        only 'probs' is matter
        """
        idx_tta = [[-1], [-2], [-1, -2]]
        #idx_tta = [[-1]]
        #idx_tta = [[-2]]
        #idx_tta = []
        preds = self.segmentor(input_x)
        #preds['full_probs'] = F.interpolate(preds['probs'], size=self._shape, mode='bicubic')
        for idx_flip in idx_tta:
            preds_tta = self.segmentor(torch.flip(input_x, dims=idx_flip))
            x = torch.flip(preds_tta['probs'], dims=idx_flip)
            #x = F.interpolate(x, self._shape, mode='bicubic')
            preds['probs'] += x
        preds['probs'] /= 1 + len(idx_tta)
        return preds

    def validation_step(
            self,
            batch_dict: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, Any]:
        self._shape = batch_dict['full_target'].shape[-2:]
        return super(TTAHolder, self).validation_step(batch_dict, batch_idx)
