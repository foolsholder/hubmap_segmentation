import pytorch_lightning as pl
import os
import json
import argparse
import torch
import numpy as np

from collections import OrderedDict
from typing import Dict, Union
from hubmap_segmentation.holders import create_holder
from hubmap_segmentation.sdataset import create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


parser = argparse.ArgumentParser(description='Get config')
parser.add_argument('--config_path', type=str, required=True, default='')
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

config: Dict[str, Union[Dict, str, int]] = json.load(open(args.config_path, 'r'), object_pairs_hook=OrderedDict)

FOLD: int = args.fold

devices = torch.cuda.device_count()
config['wandb_cfg']['name'] = config['wandb_cfg']['name'].format(
    os.environ['DEVICE_TYPE'],
    devices,
    FOLD
)
config['train_loader']['fold'] = FOLD
config['valid_loader']['fold'] = FOLD

max_epochs: int = config['max_epochs']
if 'seed' in config.keys():
    if config['seed']:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

holder_cfg = config.pop('holder_cfg')
holder_cfg['config'] = config
if 'type' in holder_cfg and holder_cfg['type'] != 'base':
    holder_cfg['model_type'] = config['model_cfg']['type']
model_holder = create_holder(holder_cfg)
wandb_logger = WandbLogger(
    **config['wandb_cfg'],
    config=config
)
trainer = pl.Trainer(
    max_epochs=max_epochs,
    strategy=config['strategy'],
    accelerator='gpu',
    devices=devices,
    num_nodes=config['num_nodes'],
    sync_batchnorm=True,
    log_every_n_steps=config['log_every_n_steps'],
    gradient_clip_val=config['gradient_clip_val'],
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(
                os.environ['SHUBMAP_EXPS'],
                config['wandb_cfg']['name']
            ),
            save_weights_only=True,
            save_top_k=10,
            monitor='dice/wo_lung',
            mode='max'
        ),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(
                os.environ['SHUBMAP_EXPS'],
                config['wandb_cfg']['name']
            ),
            save_weights_only=True,
            save_last=True,
            save_top_k=-0,
        )
    ],
    logger=wandb_logger
)

trainer.fit(
    model=model_holder,
    train_dataloaders=create_loader_from_cfg(config['train_loader']),
    val_dataloaders=create_loader_from_cfg(config['valid_loader']),
)
