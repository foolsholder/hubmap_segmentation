import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class SigmoidSoftDiceLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'sigmoid_soft_dice',
    ):
        super().__init__(loss_name=loss_name)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        probs = preds['probs']
        target = batch['target']

        batch_size = target.shape[0]
        probs = probs.view(batch_size, -1)
        target = target.view(batch_size, -1)

        name = self._name

        mult = (2 * probs * target).sum(dim=1) + 1.
        denom = probs.sum(dim=1) + target.sum(dim=1) + 1.
        dice_loss = 1 - mult / denom
        dice_loss = torch.mean(dice_loss)
        return name, dice_loss
