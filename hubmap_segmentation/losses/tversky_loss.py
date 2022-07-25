import torch

import pytorch_lightning as pl
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F


class TverskyLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'tversky_loss',
            alpha: float = 0.7,
            gamma: float = 0.75,
            focal: bool = False
    ):
        super().__init__(loss_name=loss_name)
        self.gamma = gamma
        self.alpha = alpha
        self.focal = focal

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

        tp = (probs * target).sum() +1.
        fn = ((1-probs) * target).sum() +1.
        fp = (probs * (1-target)).sum() +1.
        denom = tp + self.alpha*fn + (1-self.alpha)*fp + 1.
        tversky_loss = 1 - tp / denom
        tversky_loss = torch.mean(tversky_loss)
        if self.focal:
            return name, torch.pow(tversky_loss, self.gamma)
        return name, tversky_loss
