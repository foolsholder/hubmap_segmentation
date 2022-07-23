import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class LossAggregation(LossMetric):
    def __init__(
            self,
            weights: Dict[str, float],
            loss_name: str = 'total_loss',
    ):
        super().__init__(loss_name=loss_name)
        self._weights = weights

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        total_loss = None
        for k, w in self._weights.items():
            if total_loss is None:
                total_loss = w * preds[k]
            else:
                total_loss += w * preds[k]

        name = self._name
        return name, total_loss
