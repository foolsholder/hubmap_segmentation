import pytorch_lightning as pl
import os
import torch

from hubmap_segmentation.holders import EnsembleHolder
from hubmap_segmentation.sdataset import create_loader

config = {
    'model_cfg': {
        'type': 'swin',
        'size': 'small',
        'load_weights': ''
    },
    "holder_cfg": {
        "tiling_height": 1024,
        "tiling_width": 1024,
        "use_tiling_inf": True
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
            '1024',
            'epoch.ckpt'
        )
    ],
    weights=weights,
    tta_list=[
        ('flip', [-1]),
        ('flip', [-2]),
        ('flip', [-1, -2]),
        #('transpose', None),
        #('rotate90', 1),
        #('rotate90', 2),
        #('rotate90', 3),
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
        height=1024,
        width=1024,
        fold=0
    )
)

print('Max memory allocated {:.2f} MB'.format(torch.cuda.max_memory_allocated('cuda:0') / 1024./ 1024.))
