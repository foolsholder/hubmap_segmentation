import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from holder import ModelHolder
from sdataset import create_loader, create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'tom',
        'load_weights': ''
    },
    'wandb_cfg': {
        'project': 'hubmap',
        'name': 'focal_bce+soft_dice_tom'
    },
    'train_loader': {
        'train': True,
        'batch_size': 1,
        'num_workers': 4,
        'height': 1024,
        'width': 1024
    },
    'valid_loader': {
        'train': False,
        'batch_size': 1,
        'num_workers': 4,
        'height': 1024,
        'width': 1024
    }
}

model_holder = ModelHolder(config)
wandb_logger = WandbLogger(**config['wandb_cfg'])
trainer = pl.Trainer(
    min_epochs=100,
    accelerator='ddp',
    gpus=4,
    num_nodes=1,
    log_every_n_steps=1,
    weights_save_path=os.environ['SHUBMAP_EXPS'],
    gradient_clip_val=1.0,
    callbacks=[pl.callbacks.LearningRateMonitor(logging_interval='step')],
    logger=wandb_logger
)

trainer.fit(
    model=model_holder,
    train_dataloaders=create_loader_from_cfg(config['train_loader']),
    val_dataloaders=create_loader_from_cfg(config['valid_loader']),
)
