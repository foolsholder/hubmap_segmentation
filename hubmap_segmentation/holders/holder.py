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
from hubmap_segmentation.holders.optimizer_utils import create_opt_shed
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


class ModelHolder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any],
            tiling_height: Optional[int] = None,
            tiling_width: Optional[int] = None,
            use_tiling_inf: bool = False,
            smooth: float = 1e-7,
            thr: float = 0.5
    ):
        super(ModelHolder, self).__init__()
        self._config = deepcopy(config)

        self._num_classes = config['model_cfg']['num_classes']

        self.segmentor: torch.nn.Module = create_model(config['model_cfg'])

        self.tiling_height: int = tiling_height
        self.tiling_width: int = tiling_width
        self.use_tiling_inf: bool = use_tiling_inf

        metrics = [Dice(thr, smooth)]

        self.metrics_names = []
        for metric in metrics:
            self.__setattr__(metric._name, metric)
            self.metrics_names += [metric._name]

        self._stages_names = ['train', 'valid']
        self.losses_names = []
        self.aux_losses_names = []

        if 'losses' in config:
            losses = [
                *[available_losses[loss]() for loss in config['losses']['names']],
            ]

            if 'aux' in config['losses'] \
                    and config['losses']['aux'] > 0. \
                    and len(losses) > 0:
                aux_losses = [
                    *[available_losses[loss](aux=True) for loss in config['losses']['names']],
                ]
                aux_weight = config['losses'].pop('aux')

                loss_names = copy(config['losses']['names'])
                loss_weigths = copy(config['losses']['weights'])

                for loss_name, loss_weigth in zip(loss_names, loss_weigths):
                    config['losses']['names'] += ['aux_' + loss_name]
                    config['losses']['weights'] += [loss_weigth * aux_weight]

                losses += aux_losses
            losses += [
                LossAggregation(
                    dict(zip(config['losses']['names'], config['losses']['weights'])),
                    loss_name='loss'
                )
            ]

            for loss in losses:
                self.__setattr__(loss._name, loss)
                self.losses_names += [loss._name]

    def training_step(
            self,
            batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        return self._step_logic(batch_dict, 'train')

    def _step_logic(
            self,
            batch_dict: Dict[str, torch.Tensor],
            stage: str
    ) -> Dict[str, Any]:
        input_x = batch_dict['input_x']

        preds: Dict[str, torch.Tensor] = self.forward(input_x, additional_info=batch_dict, stage=stage)

        if stage != 'valid':
            for loss_name in self.losses_names:
                loss = self.__getattr__(loss_name)
                dct = loss.calc_loss_and_update_state(
                    preds,
                    batch_dict,
                    stage=stage
                )
                for k, v in dct.items():
                    self.log('{}_batch/{}'.format(k, stage), v, prog_bar=True, sync_dist=True)
                preds.update(dct)
        return preds

    def validation_step(
            self,
            batch_dict: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, Any]:
        preds = self._step_logic(batch_dict, 'valid')
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
                self.log(k, v, prog_bar=True, sync_dist=True)
            metric.reset()
        self.log_and_reset_losses()

    def log_and_reset_losses(self) -> None:
        for loss_name in self.losses_names:
            loss = self.__getattr__(loss_name)
            for stage in self._stages_names:
                if stage == 'valid':
                    continue
                dct = loss.compute_loader_and_name(stage)
                for k, v in dct.items():
                    self.log(k, v, prog_bar=True, sync_dist=True)
            loss.reset()

    def _forward_impl(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        preds = self.segmentor(input_x, additional_info)
        return preds

    def _forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._forward_impl(input_x, additional_info)

    def forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None,
            stage: str = 'valid'
    ) -> Dict[str, Any]:
        if stage != 'valid' or not self.use_tiling_inf:
            preds: Dict[str, torch.Tensor] = self._forward(input_x, additional_info)
        else:
            preds: Dict[str, torch.Tensor] = self.sliding_window(input_x, additional_info)
        return preds

    def configure_optimizers(self):
        return create_opt_shed(self._config['opt_sched'], self.segmentor.parameters())

    def sliding_window(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if 'full_image' in additional_info:
            input_x = additional_info['full_image']

        input_x, (pad_h, pad_w) = self._pad_if_need(input_x)

        channels = self._num_classes
        h, w = input_x.shape[2:]

        shift_h = self.tiling_height - self.tiling_height // 2
        shift_w = self.tiling_width - self.tiling_width // 2

        weight = torch.zeros((1, h, w)).to(input_x.device)
        probs = torch.zeros((channels, h, w)).to(input_x.device)

        # logits
        # probs

        h_cnt = max(0, h - self.tiling_height + shift_h - 1) // shift_h + 1
        w_cnt = max(0, w - self.tiling_width + shift_w - 1) // shift_w + 1

        #print(h_cnt, w_cnt, shift_h, shift_w, h, w, flush=True)

        for h_idx in range(h_cnt):
            h_right = min(h, shift_h * h_idx + self.tiling_height)
            h_left = h_right - self.tiling_height
            for w_idx in range(w_cnt):
                w_right = min(w, shift_w * w_idx + self.tiling_width)
                w_left = w_right - self.tiling_width

                weight[:, h_left:h_right, w_left:w_right] += 1
                input_window = input_x[:, :, h_left:h_right, w_left:w_right]

                preds = self._forward(input_window, additional_info)
                # [1, C, T_H, T_W]
                window_probs = preds['probs'][0] # [C; T_H; T_W]

                probs[:, h_left:h_right, w_left:w_right] += window_probs
        probs = (probs / weight)[None]
        probs = self._delete_pad_if_need(probs, pad_h, pad_w)
        return {
            "probs": probs
        }

    def _pad_if_need(self, input_x: torch.Tensor):
        h, w = input_x.shape[2:]
        pad_h = 0
        pad_w = 0
        if h < self.tiling_height:
            pad_h = self.tiling_height - h
        if w < self.tiling_width:
            pad_w = self.tiling_width - w
        input_x = F.pad(input_x, pad=(
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2
        ))
        return (input_x, (pad_h, pad_w))

    def _delete_pad_if_need(self, input_x: torch.Tensor,
                            pad_h: int, pad_w: int)  -> torch.Tensor:
        h, w = input_x.shape[2:]
        top_h = pad_h // 2
        bot_h = pad_h - pad_h // 2
        left_w = pad_w // 2
        right_w = pad_w - left_w
        return input_x[:,:, top_h:h-bot_h, left_w:w-right_w]
