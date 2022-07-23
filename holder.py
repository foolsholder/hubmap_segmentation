import torch
import torchmetrics
import wandb
import pl_bolts

import pytorch_lightning as pl

from torch.nn import functional as F
from typing import (
    Dict, Optional, List, Tuple,
    Callable, Union, Any, Sequence
)
from callbacks.dice_metric import Dice

from models.utils import create_model


class ModelHolder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        super(ModelHolder, self).__init__()
        self.segmentor = create_model(config['model_cfg'])

        self.dice = Dice(0.5, 1e-7)

    def training_step(
            self,
            batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        preds = self._step_logic(batch_dict)
        target = batch_dict['target']
        logits = preds['logits']
        probs = preds['probs']
        #loss = F.binary_cross_entropy_with_logits(logits, target)
        mult = (2 * probs * target).view(probs.size(0), -1).sum(dim=1)
        probs = probs.view(probs.size(0), -1).sum(dim=1)
        target = target.view(probs.size(0), -1).sum(dim=1)
        loss = 1 - torch.mean((mult+ self.smooth) / (probs + target))
        preds['soft_dice'] = loss

        target = batch_dict['target']
        bce = F.binary_cross_entropy_with_logits(logits, target)
        preds['bce'] = bce

        self.log('soft_dice', preds['soft_dice'], prog_bar=True)
        self.log('bce', preds['bce'], prog_bar=True)
        preds['loss'] = 0.2 * preds['bce'] + preds['soft_dice']
        return preds


    def _step_logic(
            self,
            batch_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        input_x = batch_dict['input_x']
        target = batch_dict['target']
        preds = self.forward(input_x)
        return preds

    def validation_step(
            self,
            batch_dict: Dict[str, torch.Tensor],
            batch_idx: int
    ) -> Dict[str, Any]:
        preds = self._step_logic(batch_dict)
        logits = preds['logits']
        probs = preds['probs']
        target = batch_dict['target']
        #torch.save(probs.cpu(), 'probs.pth')
        #torch.save(target.cpu(), 'target.pth')
        self.dice.update(logits, target)

    def validation_epoch_end(
            self,
            outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ) -> None:
        self.log('valid_dice', self.dice.compute(), prog_bar=True)

    def forward(self, input_x: torch.Tensor) -> Dict[str, Any]:
        logits = self.segmentor(input_x)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits)
        }

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.segmentor.parameters(),
            lr=5e-3,
            weight_decay=1e-3
        )
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optim,
            warmup_epochs=20,
            max_epochs=int(1e8)
        )
        return [optim], [scheduler]
