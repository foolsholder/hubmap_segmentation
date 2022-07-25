import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from hubmap_segmentation.holder import TTAHolder
from hubmap_segmentation.sdataset import create_loader
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
        height=512,
        width=512
    ),
    ckpt_path=os.path.join(
        os.environ['SHUBMAP_EXPS'],
        'focal_bce+soft_dice_tom_imagenet_512_titan4',
        'epoch=49-step=8750.ckpt'
    )
)
