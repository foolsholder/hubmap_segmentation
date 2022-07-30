import pytorch_lightning as pl
import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from hubmap_segmentation.holder import ModelHolder
from hubmap_segmentation.sdataset import create_loader, create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger

parser = argparse.ArgumentParser(description='Get config')
parser.add_argument('--config_path', type=str, required=False, default='')
args = parser.parse_args()
if args.config_path:
    config = json.load(open(args.config_path, 'r'))
else:
    config = {
        'model_cfg': {
            'type': 'effnet',
            'load_weights': 'imagenet'
        },
        'wandb_cfg': {
            'project': 'hubmap',
            'name': 'ufocal_effnet_imagenet_512_3090'
        },
        'train_loader': {
            'train': True,
            'batch_size': 11,
            'num_workers': 8,
            'height': 512,
            'width': 512
        },
        'valid_loader': {
            'train': False,
            'batch_size': 1,
            'num_workers': 4,
            'height': 512,
            'width': 512
        },
        'losses': {
            'weights': [0.5, 1.0],
            'names': [
                'lovasz_hinge_loss',
                'unified_focal_loss'
            ]
        },
        'seed': False
    }

if 'seed' in config.keys():
    if config['seed']:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

model_holder = ModelHolder(config)
wandb_logger = WandbLogger(**config['wandb_cfg'])
trainer = pl.Trainer(
    min_epochs=100,
    strategy='ddp',
    gpus=4,
    num_nodes=1,
    log_every_n_steps=1,
    gradient_clip_val=1.0,
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(os.environ['SHUBMAP_EXPS'], config['wandb_cfg']['name']),
            save_top_k=2,
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
