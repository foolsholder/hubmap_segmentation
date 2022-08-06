import pytorch_lightning as pl
import os
import json
import argparse
import torch
import numpy as np

from typing import Dict, Union
from hubmap_segmentation.holders.holder import ModelHolder
from hubmap_segmentation.sdataset import create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


parser = argparse.ArgumentParser(description='Get config')
parser.add_argument('--config_path', type=str, required=True, default='')
args = parser.parse_args()

config: Dict[str, Union[Dict, str, int]] = json.load(open(args.config_path, 'r'))
max_epochs: int = config['max_epochs']
if 'seed' in config.keys():
    if config['seed']:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

model_holder = ModelHolder(config)
wandb_logger = WandbLogger(**config['wandb_cfg'])
trainer = pl.Trainer(
    max_epochs=max_epochs,
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
