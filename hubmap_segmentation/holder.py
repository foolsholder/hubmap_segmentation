import torch
import torchmetrics
import wandb
import pl_bolts

import pytorch_lightning as pl

from copy import deepcopy
from torch.nn import functional as F
from typing import (
    Dict, Optional, List, Tuple,
    Callable, Union, Any, Sequence
)
from .metrics.dice_metric import Dice
from .losses import (
    BCELoss, SigmoidSoftDiceLoss,
    LossAggregation, LossMetric,
    BinaryFocalLoss, TverskyLoss,
    LovaszHingeLoss, SymmetricUnifiedFocalLoss
)
from .models.utils import create_model

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
        preds: Dict[str, torch.Tensor] = self.forward(input_x)

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
    ) -> Dict[str, Any]:
        preds: Dict[str, torch.Tensor] = self.segmentor(input_x)
        return preds

    def configure_optimizers(self):
        optim = torch.optim.RAdam(
            self.segmentor.parameters(),
            lr=1e-4,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=75,
            warmup_start_lr=1e-7,
            eta_min=1e-7,
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
        idx_tta = [
            ('flip', [-1]),
            ('flip', [-2]),
            ('flip', [-1, -2]),
            ('transpose', None),
            ('rotate90', 1),
            ('rotate90', 2),
            ('rotate90', 3),
        ]
        batch_input = [input_x]

        #preds['full_probs'] = F.interpolate(preds['probs'], size=self._shape, mode='bicubic')
        for type_aug, args_aug in idx_tta:
            input_y = input_x
            if type_aug == 'flip':
                input_y = torch.flip(input_y, dims=args_aug)
            elif type_aug == 'transpose':
                input_y = torch.transpose(input_y, dim0=2, dim1=3)
            elif type_aug == 'rotate90':
                input_y = torch.rot90(input_y, k=args_aug, dims=[2, 3])
            batch_input += [input_y]

        batch_input = torch.cat(batch_input, dim=0)
        preds = self.segmentor(batch_input)
        preds.pop('logits')

        idx_preds = 1
        for type_aug, args_aug in idx_tta:
            x = preds['probs'][idx_preds:idx_preds+1].clone()
            if type_aug == 'flip':
                x = torch.flip(x, dims=args_aug)
            elif type_aug == 'transpose':
                x = torch.transpose(x, dim0=2, dim1=3)
            elif type_aug == 'rotate90':
                x = torch.rot90(x, k=4-args_aug, dims=[2, 3])
            #x = F.interpolate(x, self._shape, mode='bicubic')
            preds['probs'][idx_preds:idx_preds+1] = x
            idx_preds += 1

        preds['probs'] = torch.mean(preds['probs'], dim=0, keepdim=True)
        return preds


def _clear_segmentor_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        res = type(state_dict)()
        str_patt = 'segmentor.'
        for k, v in state_dict.items():
            new_k = k[len(str_patt):]
            res[new_k] = v
        return res


class EnsembleHolder(TTAHolder):
    def __init__(self, ckpt_path_list: List[str], *args, **kwargs):
        super(EnsembleHolder, self).__init__(*args, **kwargs)
        self._stages_names = ['valid']
        self.state_dicts = []
        for path in ckpt_path_list:
            state_dict = torch.load(path, map_location='cpu')
            state_dict = state_dict['state_dict']
            state_dict = _clear_segmentor_prefix(state_dict)
            self.state_dicts += [state_dict]

    def forward(self, input_x: torch.Tensor) -> Dict[str, Any]:
        probs = []

        for st in self.state_dicts:
            self.segmentor.load_state_dict(st)
            preds_tmp = super(EnsembleHolder, self).forward(input_x)
            probs += [preds_tmp['probs']]

        probs = torch.cat(probs, dim=0)
        probs = torch.mean(probs, dim=0, keepdim=True)
        preds = {
            'probs': probs
        }
        return preds
