import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from holder import TTAHolder
from sdataset import create_loader
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'tom',
        'load_weights': ''
    },
    'wandb_cfg': {
        'project': 'hubmap',
        'name': 'focal_bce+soft_dice_tom_kidney'
    }
}

model_holder = TTAHolder(config)

trainer = pl.Trainer(
    accelerator='gpu',
    log_every_n_steps=1,
)

trainer.validate(
    model=model_holder,
    dataloaders=create_loader(
        train=False,
        batch_size=1,
        num_workers=2,
        height=1024,
        width=1024
    ),
    ckpt_path=os.path.join(
        os.environ['SHUBMAP_EXPS'],
        'hubmap/28n1pttw/checkpoints',
        'epoch.ckpt'
    )
)
