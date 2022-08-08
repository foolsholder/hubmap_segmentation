import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class BinaryFocalLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'binary_focal_loss',
            gamma: float = 2.,
            alpha: float = 0.25,
            **kwargs
    ):
        super().__init__(loss_name=loss_name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        logits = preds[self.prefix + 'logits']
        probs = preds[self.prefix + 'probs']
        target = batch['target']

        name = self._name
        log_probs = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none')
        gamma = self.gamma
        weights = target * (1 - probs) ** gamma + self.alpha * (1 - target) * probs ** 2
        focal_loss = torch.mean(weights * log_probs)
        return name, focal_loss
