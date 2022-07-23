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

from models.utils import create_model


class ModelHolder(pl.LightningModule):
    def __init__(
            self,
            config: Dict[str, Any]
    ):
        super(ModelHolder, self).__init__()
        self.segmentor = create_model(config['model_cfg'])

        self.thr = 0.5
        self.smooth = 1e-7

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

    def _calculate_dice(
            self,
            logits: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        batch_size = logits.shape[0]
        probs = (logits >= 0.).view(batch_size, -1).float()
        target = (target > 0.).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        dice = (2 * mult + self.smooth) / (denom + self.smooth)
        return torch.mean(dice)

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
        dice = self._calculate_dice(logits, target)
        preds.update({
            'dice': dice,
            'batch_size': target.shape[0]
        })

    def validation_epoch_end(
            self,
            outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
    ) -> None:
        dice_valid_loader = 0
        if isinstance(outputs, dict):
            dice_valid_loader = outputs['dice'].item()
        else:
            sum = 0
            for output_valid in outputs:
                dice_valid_loader += output_valid['dice'].item() * output_valid['batch_size']
                sum += output_valid['batch_size']
            dice_valid_loader /= sum + self.smooth
        self.log('valid_dice', dice_valid_loader, prog_bar=True)

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
