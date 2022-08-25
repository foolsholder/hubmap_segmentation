import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class SigmoidSoftDiceLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'sigmoid_soft_dice',
            **kwargs
    ):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        probs = preds[self.prefix + 'probs']
        target = batch['target']

        batch_size = target.shape[0]
        if probs.shape[1] != 1:
            organ_id = batch['organ_id'].long()
            probs = probs[torch.arange(batch_size), organ_id]
            target = target[torch.arange(batch_size), organ_id]

        probs = probs.view(batch_size, -1)
        target = target.view(batch_size, -1)

        name = self._name

        mult = (2 * probs * target).sum(dim=1) + 1.
        denom = probs.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1.
        dice_loss = 1 - mult / denom
        dice_loss = torch.mean(dice_loss)
        return name, dice_loss


class CatSoftDiceLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'cat_soft_dice',
            **kwargs
    ):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        target = batch['target']
        probs = torch.softmax(logits, dim=1)

        batch_size = target.shape[0]
        if probs.shape[1] != 1:
            organ_id = batch['organ_id'].long()
            probs = probs[torch.arange(batch_size), organ_id]
            target = target[torch.arange(batch_size), organ_id]

        probs = probs.view(batch_size, -1)
        target = target.view(batch_size, -1)

        name = self._name

        mult = (2 * probs * target).sum(dim=1) + 1.
        denom = probs.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1.
        dice_loss = 1 - mult / denom
        dice_loss = torch.mean(dice_loss)
        return name, dice_loss


class NLSDLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'negative_log_cat_soft_dice',
            **kwargs
    ):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        target = batch['target']
        probs = torch.softmax(logits, dim=1)

        batch_size = target.shape[0]
        if probs.shape[1] != 1:
            organ_id = batch['organ_id'].long()
            probs = probs[torch.arange(batch_size), organ_id]
            target = target[torch.arange(batch_size), organ_id]

        probs = probs.view(batch_size, -1)
        target = target.view(batch_size, -1)

        name = self._name

        mult = (2 * probs * target).sum(dim=1) + 1e-5
        denom = probs.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + 1e-5
        log_dice_loss = torch.log(mult) - torch.log(denom)
        log_dice_loss = -torch.mean(log_dice_loss)
        return name, log_dice_loss
