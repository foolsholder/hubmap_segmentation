import torch

import pytorch_lightning as pl
from torchmetrics import Metric
from typing import Dict, Any, Optional, List
from torch.nn import functional as F


class Dice(Metric):
    def __init__(self, thr: float, smooth: float):
        super().__init__()
        self.thr = thr
        self.smooth = smooth
        self._name = 'dice'

        all = ['lung', 'spleen', 'kidney', 'prostate', 'largeintestine', 'wo_kidney']
        self.all = all
        for organ in all:
            self.add_state('{}_dice'.format(organ), default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state('{}_samples'.format(organ), default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("sum_dice", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(
            self,
            preds: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor]
    ):
        probs = preds['probs'] # ON VALID PROBS ALREADY FULL_SHAPE_TENSOR
        target = batch['full_target'] # ALWAYS 1x3000x3000
        probs = F.interpolate(probs, size=target.shape[-2:], mode='bicubic') # -> FULL_RESOLUTION

        if probs.shape[1] != 1:
            organ_id = batch['organ_id'][0].cpu().item()
            #probs = probs[:, organ_id:organ_id+1]
            probs = (torch.argmax(probs, dim=1, keepdim=True) == organ_id).long()
            #target = target[:, organ_id:organ_id+1]

        # images [1, 1, H, W]

        batch_size = probs.shape[0]
        probs = (probs >= self.thr).view(batch_size, -1).float()
        target = (target > 0.).view(batch_size, -1).float()
        mult = target * probs
        mult = torch.sum(mult, dim=1)
        denom = torch.sum(probs, dim=1) + torch.sum(target, dim=1)
        dice = (2 * mult + self.smooth) / (denom + self.smooth)

        for idx, organ in enumerate(batch['organ']):
            #attr_sum = self.__getattr__('{}_dice'.format(organ))
            #attr_sum += dice[idx]
            #attr_cnt = self.__getattr__('{}_samples'.format(organ))
            #attr_cnt += 1
            if organ == 'kidney':
                self.kidney_dice += dice[idx]#.item()
                self.kidney_samples += 1
            elif organ == 'lung':
                self.lung_dice += dice[idx]#.item()
                self.lung_samples += 1
            elif organ == 'spleen':
                self.spleen_dice += dice[idx]#.item()
                self.spleen_samples += 1
            elif organ == 'prostate':
                self.prostate_dice += dice[idx]#.item()
                self.prostate_samples += 1
            elif organ == 'largeintestine':
                self.largeintestine_dice += dice[idx]#.item()
                self.largeintestine_samples += 1
            if organ != 'kidney':
                self.wo_kidney_dice += dice[idx]
                self.wo_kidney_samples += 1

        #with open('full_res_metrics.txt', 'a') as f:
        #    f.write('{}, {}, {}\n'.format(batch['image_id'][0], batch['organ'][0], dice.item()))
        dice = torch.sum(dice)
        self.sum_dice += dice
        self.total_samples += batch_size

    def compute_every(self) -> Dict[str, torch.Tensor]:
        res = {}
        for organ in self.all:
            #attr_sum = self.__getattr__('{}_dice'.format(organ))
            #attr_cnt = self.__getattr__('{}_samples'.format(organ))
            #res[self._name + '/' + organ] = attr_sum / attr_cnt
            if organ == 'kidney':
                value = self.kidney_dice.float() / self.kidney_samples
            elif organ == 'lung':
                value = self.lung_dice.float() / self.lung_samples
            elif organ == 'spleen':
                value = self.spleen_dice.float() / self.spleen_samples
            elif organ == 'prostate':
                value = self.prostate_dice.float() / self.prostate_samples
            elif organ == 'largeintestine':
                value = self.largeintestine_dice.float() / self.largeintestine_samples
            else:
                value = self.wo_kidney_dice.float() / self.wo_kidney_samples
            res[self._name + '/' + organ] = value
        res[self._name + '/valid'] = self.compute()
        return res

    def compute(self) -> torch.Tensor:
        return self.sum_dice.float() / self.total_samples
