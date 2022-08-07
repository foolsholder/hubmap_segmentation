from pytorch_lightning.callbacks import Callback

from typing import Optional, Any
import torch.nn.functional as F
import torch
import wandb

from pytorch_lightning import Trainer,LightningModule
class MyPrintingCallback(Callback):
    def __init__(self):
        self.table = wandb.Table(columns=['ID', 'organ', 'Image', 'Dice'])
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        predicted_mask = outputs['probs']
        image_id = batch['image_id']
        organ = batch['organ']
        mask = batch['full_target']

        target = mask
        probs = predicted_mask
        probs = F.interpolate(probs, size=target.shape[-2:], mode='bicubic')  # -> FULL_RESOLUTION
        batch_size = probs.shape[0]
        probs = (probs >= 0.5).view(batch_size, -1).float()
        target = (target > 0.).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        dice = (2 * mult + 1e-7) / (denom + 1e-7)
        del target, probs, mult, denom
        mask = mask.squeeze(0).squeeze(0).cpu().numpy()
        predicted_mask = (predicted_mask >= 0.5).squeeze(0).squeeze(0).cpu().numpy()
        print(mask.shape, predicted_mask.shape)
        mask_img = wandb.Image(
            batch['full_image'],
            masks={
                "prediction": {
                    "mask_data": predicted_mask,
                },
                "ground_truth": {
                    "mask_data": 1.0-mask,
                },
            }
        )
        self.table.add_data(image_id, organ, mask_img, dice)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb.log({"Table": self.table})
        del self.table