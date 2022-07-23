import torch

import pytorch_lightning as pl

from typing import Dict, Any


class DiceMetricCallback(pl.Callback):
    def __init__(self):
        super(DiceMetricCallback, self).__init__()
        self.thr = 0.5
        self.smooth = 1e-7

    def _calculate_dice(
            self,
            probs: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        batch_size = probs.shape[0]
        probs = (probs > self.thr).view(batch_size, -1).float()
        target = (target > self.thr).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        dice = (2 * mult + self.smooth) / (denom + self.smooth)
        return dice

