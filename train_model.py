import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from holder import ModelHolder
from sdataset import create_loader, create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'tom',
        'load_weights': 'imagenet'
    },
    'wandb_cfg': {
        'project': 'hubmap',
        'name': 'focal_bce+soft_dice_tom_imagenet_512_titan4'
    },
    'train_loader': {
        'train': True,
        'batch_size': 4,
        'num_workers': 4,
        'height': 512,
        'width': 512
    },
    'valid_loader': {
        'train': False,
        'batch_size': 1,
        'num_workers': 4,
        'height': 512,
        'width': 512
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
