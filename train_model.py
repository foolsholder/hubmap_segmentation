import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from hubmap_segmentation.holder import ModelHolder
from hubmap_segmentation.sdataset import create_loader, create_loader_from_cfg
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'effnet',
        'load_weights': 'imagenet'
    },
    'wandb_cfg': {
        'project': 'hubmap',
        'name': 'fbce+sdice+tversky+hinge_effnet_imagenet_512_A4'
    },
    'train_loader': {
        'train': True,
        'batch_size': 40,
        'num_workers': 20,
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
