import pytorch_lightning as pl
import os
import torch
import numpy as np
from typing import Optional, Dict, Any

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
    "seed": 3496295,
}

if 'seed' in config.keys():
    if config['seed']:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

weights = None
aux_holders = tuple()

import random
seed = config['seed']

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model_holder = EnsembleDifferent(
    config=config,
    ckpt_path_list=[
        os.path.join(
            os.environ['SHUBMAP_EXPS'],
            'tmp',
            'epoch_effnet_f0.ckpt'
        )
    ],
    weights=weights,
    tta_list=[
        #('flip', [-2]),
        #('flip', [-1]),
        #('flip', [-1, -2]),
        #('transpose', None),
        #('rotate90', 1),
        #('rotate90', 2),
        #('rotate90', 3),
    ],
    yet_another_holders=aux_holders
)


device = 'cuda:0'
model_holder.eval().to(device)

valid_loader =create_loader(
    train=False,
    batch_size=1,
    num_workers=2,
    height=512,
    width=512,
    fold=0
)

torch.backends.cudnn.deterministic = True
t = 0
batch: Optional[Dict[str, Any]] = None
for tmp in valid_loader:
    batch = tmp


    organ_id = batch["organ_id"].item()
    print(f'image_id: {batch["image_id"][0]}, organ: {batch["organ"][0]}')
    print(f'h: {batch["full_image"].shape[2]}, w: {batch["full_image"].shape[3]}')

    batch['full_image'] = batch['full_image'].to(device)
    batch['full_target'] = batch['full_target'].to(device)

    model_holder.segmentor.eval()
    with torch.no_grad():
        preds = model_holder.validation_step(batch, batch_idx=t)
    t = t + 1
    if t == 2:
        break
    #print(f'some_values: {list(preds["probs"].cpu()[0, organ_id, 12, 13:16].numpy())}')
    #print(f'some_input_values: {list(batch["full_image"].cpu()[0, 2, 12, 13:16].numpy())}')

dice = model_holder.dice.compute()
print('dice: {:.5f}'.format(dice))

print('Max memory allocated {:.2f} MB'.format(
    torch.cuda.max_memory_allocated('cuda:0') / 1024./ 1024.
))
        #('flip', [-1]),
        #('flip', [-1, -2]),
        #('transpose', None),
        #('rotate90', 1),
        #('rotate90', 2),
        #('rotate90', 3),
