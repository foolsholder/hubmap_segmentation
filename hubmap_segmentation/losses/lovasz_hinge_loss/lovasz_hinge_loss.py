import torch

import pytorch_lightning as pl
from ..loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F
from .utils import lovasz_softmax


class LovaszHingeLoss(LossMetric):
    def __init__(
            self,
            loss_name: str = 'cat_lovasz',
            **kwargs
    ):
        super().__init__(loss_name=loss_name, **kwargs)

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        probs = preds[self.prefix + 'logits'].softmax(dim=1)#[:, 0, :, :]
        target = batch['cat_target']#[:, 0, :, :]

        name = self._name
        value = lovasz_softmax(probs, target)
        return name, value
