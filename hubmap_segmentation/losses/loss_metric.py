import torch

import pytorch_lightning as pl
from torchmetrics import Metric
from typing import Dict, Any, Tuple


class LossMetric(Metric):
    def __init__(self, loss_name: str, **kwargs):
        super(LossMetric, self).__init__(**kwargs)
        self._name = loss_name

        self.add_state("train_sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("train_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.add_state("valid_sum_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("valid_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, *_: Any, **__: Any) -> None:
        pass

    def compute(self) -> Any:
        pass

    def batch_loss_and_name(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = ''
    ) -> Tuple[str, torch.Tensor]:
        pass

    def calc_loss_and_update_state(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            stage: str = ''
    ) -> Dict[str, torch.Tensor]:
        name, value = self.batch_loss_and_name(preds, batch, stage)
        batch_size: int = int(batch['target'].shape[0])
        if stage == 'valid':
            self.valid_sum_loss += value.item() * batch_size
            self.valid_samples += batch_size
        else:
            self.train_sum_loss += value.item() * batch_size
            self.train_samples += batch_size
        return {
            name: value
        }

    def compute_and_name(self, stage: str = 'train') -> Tuple[str, torch.Tensor]:
        name = '{}_loader/{}'.format(self._name, stage)
        if stage == 'valid':
            return name, self.valid_sum_loss / self.valid_samples
        else:
            return name, self.train_sum_loss / self.train_samples

    def compute_loader_and_name(self, stage: str = 'train') -> Dict[str, torch.Tensor]:
        name, value = self.compute_and_name(stage)
        return {
            name: value
        }
