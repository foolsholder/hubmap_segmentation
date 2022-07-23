import torch

import pytorch_lightning as pl
from torchmetrics import Metric
from typing import Dict, Any

class Dice(Metric):
    def __init__(self, thr: float, smooth: float):
        super().__init__()
        self.thr = thr
        self.smooth = smooth
        self.add_state("batched_dice", default=torch.Tensor(0).float())
        self.add_state("total_batches", default=torch.Tensor(0))

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        print(logits.shape, target.shape)
        batch_size = logits.shape[0]
        probs = (logits >= 0.).view(batch_size, -1).float()
        target = (target > 0.).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        print(mult, denom)
        self.batched_dice += torch.mean((2 * mult + self.smooth) / (denom + self.smooth))
        self.total_batches += 1.0
        print("DICE info:", self.batched_dice, self.total_batches)

    def compute(self) -> torch.Tensor:
        return self.batched_dice.float() / self.total_batches