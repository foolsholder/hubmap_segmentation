import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class BCELoss(LossMetric):
    def __init__(self, loss_name: str = 'bce', **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        target = batch['target']

        name = self._name
        value = F.binary_cross_entropy_with_logits(logits, target)
        return name, value
