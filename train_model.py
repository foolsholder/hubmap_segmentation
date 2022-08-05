import pytorch_lightning as pl
import os
import json
import argparse
import torch
import numpy as np

from typing import Dict, Union
from torch.utils.data import DataLoader
from hubmap_segmentation.holder import ModelHolder
from hubmap_segmentation.sdataset import create_loader, create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


parser = argparse.ArgumentParser(description='Get config')
parser.add_argument('--config_path', type=str, required=False, default='')
args = parser.parse_args()
if args.config_path:
    config: Dict[str, Union[Dict, str]] = json.load(open(args.config_path, 'r'))
else:
    config = {
        'model_cfg': {
            'type': 'tom',
            'load_weights': 'kidney_0'
        },
        'wandb_cfg': {
            'project': 'hubmap',
            'name': 'fbce+sdice_tom_kidney_radam_512_T4_F0'
        },
        'train_loader': {
            'train': True,
            'batch_size': 4,
            'num_workers': 8,
            'height': 512,
            'width': 512,
            'fold': 0
        },
        'valid_loader': {
            'train': False,
            'batch_size': 1,
            'num_workers': 4,
            'height': 512,
            'width': 512,
            'fold': 0
        },
        'losses': {
            'weights': [1.0, 1.0],
            'names': [
                'binary_focal_loss',
                'sigmoid_soft_dice'
            ]
        },
        'seed': 3496295
    }

if 'seed' in config.keys():
    if config['seed']:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

model_holder = ModelHolder(config)
wandb_logger = WandbLogger(**config['wandb_cfg'])
trainer = pl.Trainer(
    max_epochs=150,
    strategy='ddp',
    gpus=4,
    num_nodes=1,
    log_every_n_steps=1,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(
                os.environ['SHUBMAP_EXPS'],
                config['wandb_cfg']['name']
            ),
            save_weights_only=True,
            save_top_k=10,
            monitor='dice/valid',
            mode='max'
        )
    ],
    logger=wandb_logger
)

trainer.fit(
    model=model_holder,
    train_dataloaders=create_loader_from_cfg(config['train_loader']),
    val_dataloaders=create_loader_from_cfg(config['valid_loader']),
)
