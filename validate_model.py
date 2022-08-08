import pytorch_lightning as pl
import os

from hubmap_segmentation.holders import EnsembleHolder
from hubmap_segmentation.sdataset import create_loader

config = {
    'model_cfg': {
        'type': 'swin',
        'size': 'small',
        'load_weights': ''
    }
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
            'fbce+sdice_swinS_frog_adamw_512_V4_F0_SA',
            'epoch.ckpt'
        )
    ],
    weights=weights,
    tta_list=[
        ('flip', [-1]),
        ('flip', [-2]),
        ('flip', [-1, -2]),
        ('transpose', None),
        ('rotate90', 1),
        ('rotate90', 2),
        ('rotate90', 3),
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
        width=512,
        fold=0
    )
)
