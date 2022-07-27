import torch
import torch.nn as nn
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F
from .tversky_loss import TverskyLoss

class SymmetricUnifiedFocalLoss(LossMetric):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(
            self,
            loss_name: str = 'unified_focal_loss',
            weight: float = 0.5,
            delta: float = 0.6,
            gamma: float = 0.5
    ):
        super().__init__(loss_name=loss_name)
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = 'train'
    ) -> Tuple[str, torch.Tensor]:
        probs = preds['probs']
        target = batch['target']

        name = self._name
        tp = (probs * target).sum(dim=1)
        fn = ((1 - probs) * target).sum(dim=1)
        fp = (probs * (1 - target)).sum(dim=1)
        denom = tp + self.delta * fn + (1 - self.delta) * fp
        tversky_loss = torch.mean(1 - (tp + 1.) / (denom + 1.))

        symmetric_fl = torch.mean(self.delta*torch.pow(1-probs, 1-self.gamma)*F.binary_cross_entropy(probs, target))
        if self.weight is not None:
            return name, (self.weight * tversky_loss) + ((1-self.weight) * symmetric_fl)
        else:
            return name, tversky_loss + symmetric_fl
