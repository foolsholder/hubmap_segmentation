import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class DistillLoss(LossMetric):
    def __init__(self, loss_name: str = 'distill_logits_mse', **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        target = batch[self.prefix + 'logits' + '_teacher']
        name = self._name
        value = F.mse_loss(logits, target)
        return name, value
