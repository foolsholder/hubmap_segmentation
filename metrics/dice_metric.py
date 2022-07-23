import torch

import pytorch_lightning as pl
from torchmetrics import Metric
from typing import Dict, Any


class Dice(Metric):
    def __init__(self, thr: float, smooth: float):
        super().__init__()
        self.thr = thr
        self.smooth = smooth
        self.name = 'dice'
        self.add_state("sum_dice", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor]
    ):
        probs = preds['probs']
        target = batch['target']
        batch_size = probs.shape[0]
        probs = (probs >= self.thr).view(batch_size, -1).float()
        target = (target > 0.).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        self.sum_dice += torch.sum((2 * mult + self.smooth) / (denom + self.smooth))
        self.total_samples += batch_size

    def compute(self) -> torch.Tensor:
        return self.sum_dice.float() / self.total_samples
