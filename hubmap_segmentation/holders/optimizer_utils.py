import torch
import pl_bolts

from copy import copy
from typing import Dict, Union
from torch.optim import Optimizer

available_optimizers = {
    'RAdam': torch.optim.RAdam,
    'AdamW': torch.optim.AdamW,
}

available_schedulers = {
    'LinearWarmupCosineAnnealingLR': pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR
}


def create_obj(cfg, available, **kwargs):
    cfg = copy(cfg)
    type = cfg.pop('type')
    cls = available[type]
    return cls(**kwargs, **cfg)


def create_opt_shed(opt_sched_config, params) -> Dict[str, Union[Optimizer, "LRScheduler"]]:
    optim = create_obj(opt_sched_config['opt'], available_optimizers, params=params)
    res = {
        "optimizer": optim
    }
    if 'sched' in opt_sched_config:
        sched_cfg = opt_sched_config['sched']
        interval = 'epoch'
        if 'interval' in sched_cfg:
             interval = sched_cfg.pop('interval')
        scheduler = create_obj(sched_cfg, available_schedulers, optimizer=optim)

        res['lr_scheduler'] = {
            'scheduler': scheduler,
            'interval': interval
        }
    return res
