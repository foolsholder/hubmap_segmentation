import pytorch_lightning as pl
import os
import torch

from hubmap_segmentation.holders import EnsembleDifferent, EnsembleHolder
from hubmap_segmentation.sdataset import create_loader

config = {
    'model_cfg': {
        "backbone_cfg": {
          "type": "effnet",
          "load_weights": "imagenet"
        },
        "use_aux_head": False,
        "num_classes": 6
    },
    "holder_cfg": {
        "tiling_height": 512,
        "tiling_width": 512,
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

if False:
    effnet_holder = EnsembleHolder(
        config={
            "backbone_cfg": {
              "type": "effnet",
              "load_weights": "imagenet"
            },
            "use_aux_head": true,
            "num_classes": 6
        },
        ckpt_path_list=[
            os.path.join(
                os.environ['SHUBMAP_EXPS'],
                'scaled_1024_512',
                'epoch_effnet.ckpt'
            )
        ],
        weights=None,
        tta_list=[('flip', [-2])]
    )
    aux_holders = (effnet_holder,)
else:
    aux_holders = tuple()

model_holder = EnsembleDifferent(
    config=config,
    ckpt_path_list=[
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'tmp',
            'epoch_effnet_f0_cat_100.ckpt'
        )
    ],
    weights=weights,
    tta_list=[
        ('flip', [-2]),
        #('flip', [-1]),
        #('flip', [-1, -2]),
        #('transpose', None),
        ('rotate90', 1),
        #('rotate90', 2),
        #('rotate90', 3),
    ],
    yet_another_holders=aux_holders
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

print('Max memory allocated {:.2f} MB'.format(
    torch.cuda.max_memory_allocated('cuda:0') / 1024./ 1024.
))
        #('flip', [-1]),
        #('flip', [-1, -2]),
        #('transpose', None),
        #('rotate90', 1),
        #('rotate90', 2),
        #('rotate90', 3),
