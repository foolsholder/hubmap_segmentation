import torch

import pytorch_lightning as pl
from ..loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F
from .utils import lovasz_hinge


class LovaszHingeLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'lovasz_hinge_loss',
    ):
        super().__init__(loss_name=loss_name)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds['logits'][:, 0, :, :]
        target = batch['target'][:, 0, :, :]

        name = self._name
        value = lovasz_hinge(logits, target)
        return name, value
