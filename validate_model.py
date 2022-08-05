import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from hubmap_segmentation.holder import EnsembleHolder
from hubmap_segmentation.sdataset import create_loader
from pytorch_lightning.loggers import WandbLogger


config = {
    'model_cfg': {
        'type': 'tom',
        'load_weights': ''
    }
}

model_holder = EnsembleHolder(
    config=config,
    ckpt_path_list=[
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_kidney_radam_512_T4_F0',
            'epoch=119-step=2160.ckpt'
        )
    ]
)

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
    )
)
