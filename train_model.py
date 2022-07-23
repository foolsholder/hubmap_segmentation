import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from holder import ModelHolder
from sdataset import create_loader
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'simple'
    },
    'wandb_cfg': {
        'project': 'hubmap'
    }
}

model_holder = ModelHolder(config)
wandb_logger = WandbLogger(project=config['wandb_cfg']['project'])
trainer = pl.Trainer(
    min_epochs=100,
    accelerator='ddp',
    gpus=4,
    num_nodes=1,
    log_every_n_steps=1,
    weights_save_path=os.environ['SHUBMAP_EXPS'],
    logger=wandb_logger
)

trainer.fit(
    model=model_holder,
    train_dataloaders=create_loader(train=True),
    val_dataloaders=create_loader(train=False),
)
