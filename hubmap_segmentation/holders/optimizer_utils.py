import torch
import pl_bolts

from copy import copy

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


def create_opt_shed(opt_sched_config, params):
    optim = create_obj(opt_sched_config['opt'], available_optimizers, params=params)
    res = (optim,)
    if 'sched' in opt_sched_config:
        scheduler = create_obj(opt_sched_config['sched'], available_schedulers, optimizer=optim)
        res += (scheduler,)
    return res
