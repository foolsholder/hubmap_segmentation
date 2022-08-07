import pytorch_lightning as pl
import os

from hubmap_segmentation.holders import EnsembleHolder
from hubmap_segmentation.sdataset import create_loader
from hubmap_segmentation.callbacks import MyPrintingCallback
from pytorch_lightning.loggers import WandbLogger

config = {
    'model_cfg': {
        'type': 'swin',
        'size': 'small',
        'load_weights': ''
    },
    'log': False,
    "wandb_cfg": {
        "project": "hubmap_test",
        "name": "fbce+sdice_swinS_frog_adamw_512_T2_F0_SA"
    },
}

if False:
    def str2w(s, deg=2.25):
        return [float(x)**deg for x in s.split()]

    weights = [
        str2w("0.93754	0.87534	0.22033	0.77771	0.75198")
    ]
else:
    weights = None

model_holder = EnsembleHolder(
    config=config,
    ckpt_path_list=[
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'tiled_fbce+sdice_swinS_frog_adamw_512_V4_F0_SA',
            'swinS_tiled_epoch_f0.ckpt'
        )
    ],
    weights=weights,
    tta_list=[
    ]
)
if config['log']:
    wandb_logger = WandbLogger(**config['wandb_cfg'])
    trainer = pl.Trainer(
        accelerator='gpu',
        log_every_n_steps=1,
        callbacks=[MyPrintingCallback()],
        logger=wandb_logger
    )
else:
    trainer = pl.Trainer(
        accelerator='gpu',
        log_every_n_steps=1,
    )

trainer.validate(
    model=model_holder,
    dataloaders=create_loader(
        train=False,
        test=True,
        batch_size=1,
        num_workers=2,
        height=512,
        width=512
    )
)
