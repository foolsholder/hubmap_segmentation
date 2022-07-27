import torch
import torch.nn as nn
from .loss_metric import LossMetric
from typing import Dict, Any, Tuple
from torch.nn import functional as F

# Helper function to enable loss function to be flexibly used for
# both 2D or 3D image segmentation - source: https://github.com/frankkramer-lab/MIScnn

def identify_dim(shape):
    # Three dimensional
    if len(shape) == 5:
        return [1, 2, 3]

    # Two dimensional
    elif len(shape) == 4:
        return [1, 2]

    # Exception - Unknown
    else:
        raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        ce = torch.pow(1 - y_pred, self.gamma) * cross_entropy
        ce = (1 - self.delta) * ce
        loss = torch.mean(ce)
        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """

    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        dim = identify_dim(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=dim)
        fn = torch.sum(y_true * (1 - y_pred), dim=dim)
        fp = torch.sum((1 - y_true) * y_pred, dim=dim)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        dice = (1 - dice_class) * torch.pow(1 - dice_class, -self.gamma)
        # Average class scores
        loss = torch.mean(dice)
        return loss

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
        symmetric_ftl = SymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(probs, target)
        symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(probs, target)
        if self.weight is not None:
            return name, (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)
        else:
            return name, symmetric_ftl + symmetric_fl
