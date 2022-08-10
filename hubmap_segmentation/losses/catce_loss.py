import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class CATCELoss(LossMetric):
    def __init__(self, loss_name: str = 'catce', **kwargs):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        target = batch['cat_target']

        # C = 6
        weights = torch.FloatTensor([
            0.2, 1., 1., 1., 1., 1.
        ]).to(logits.device)

        name = self._name
        value = F.cross_entropy(logits, target, weight=weights)
        return name, value
