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

if True:
    def str2w(s, deg=2.25):
        return [float(x)**deg for x in s.split()]

    weights = [
        str2w("0.93754	0.87534	0.22033	0.77771	0.75198"),
        str2w("0.93928	0.88106	0.18011	0.77203	0.73264"),
        str2w("0.93495	0.87896	0.21672	0.77335	0.69706"),
        str2w("0.93220	0.87150	0.27314	0.77776	0.77479"),

        str2w("0.93955	0.88233	0.31299	0.78682	0.67036"),
        str2w("0.94067	0.88152	0.13183	0.70272	0.73217"),
        str2w("0.93148	0.87771	0.16696	0.73998	0.63653"),
        str2w("0.93876	0.86716	0.31151	0.70343	0.79625"),
    ]
else:
    weights = None

model_holder = EnsembleHolder(
    config=config,
    ckpt_path_list=[
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_kidney_radam_512_T4_F0',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_kidney_radam_512_T4_F1',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_kidney_radam_512_T4_F2',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_kidney_radam_512_T4_F3',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_imagenet_radam_512_T4_F0',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_imagenet_radam_512_T4_F1',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_imagenet_radam_512_T4_F2',
            'epoch.ckpt'
        ),
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'fbce+sdice_tom_imagenet_radam_512_T4_F3',
            'epoch.ckpt'
        ),
    ],
    weights=weights
)

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
