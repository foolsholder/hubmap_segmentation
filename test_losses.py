import json
import subprocess
from pathlib import Path
config = {
        'model_cfg': {
            'type': 'effnet',
            'load_weights': 'imagenet'
        },
        'wandb_cfg': {
            'project': 'hubmap',
            'name': 'ufocal_effnet_imagenet_512_3090'
        },
        'train_loader': {
            'train': True,
            'batch_size': 11,
            'num_workers': 8,
            'height': 512,
            'width': 512
        },
        'valid_loader': {
            'train': False,
            'batch_size': 1,
            'num_workers': 4,
            'height': 512,
            'width': 512
        },
        'losses': {
            'weights': [0.5, 1.0],
            'names': [
                'lovasz_hinge_loss',
                'unified_focal_loss'
            ]
        },
        'seed': False
    }

losses = [
    {
        'weights': [0.5, 1.0],
        'names': [
            'lovasz_hinge_loss',
            'unified_focal_loss'
        ]
    },
    {
        'weights': [1.0],
        'names': [
            'unified_focal_loss'
        ]
    },
    {
        'weights': [1.0, 1.0],
        'names': [
            'binary_focal_loss',
            'sigmoid_soft_dice'
        ]
    },
    {
        'weights': [1.0, 1.0, 0.5],
        'names': [
            'binary_focal_loss',
            'sigmoid_soft_dice',
            'lovasz_hinge_loss',
        ]
    },
]

wandb_cfgs = [
    {
         'project': 'test_losses',
         'name': 'ufocal+hinge_effnet_imagenet_512_T4'
     },
    {
         'project': 'test_losses',
         'name': 'ufocal_effnet_imagenet_512_T4'
     },
    {
         'project': 'test_losses',
         'name': 'fbce+sdice_effnet_imagenet_512_T4'
     },
    {
        'project': 'test_losses',
        'name': 'fbce+sdice_effnet_imagenet_512_T4'
    },
    {
        'project': 'test_losses',
        'name': 'fbce+sdice+hinge_effnet_imagenet_512_T4'
    },
]

seed = 3496295
config_dir = Path('configs')
try:
    config_dir.mkdir(parents=True)
except:
    pass

for loss, wandb_cfg in zip(losses, wandb_cfgs):
    exp_config = config
    exp_config['losses'] = loss
    exp_config['wandb_cfg'] = wandb_cfg
    exp_config['seed'] = seed
    config_path = Path(config_dir, exp_config['wandb_cfg']['name']+'.json')
    json.dump(exp_config, open(str(config_path), 'w'))
    try:
        p = subprocess.Popen('python train_model.py --config_path '+str(config_path), stdout=subprocess.PIPE, text=True, shell=True)
        for line in p.stdout:
            print(line)
    except:
        print("Experiment failed :(")