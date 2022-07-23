import pytorch_lightning as pl
import os

from torch.utils.data import DataLoader
from holder import ModelHolder
from sdataset import create_loader

config = {
    'model_cfg': {
        'type': 'simple'
    }
}

model_holder = ModelHolder(config)

trainer = pl.Trainer(
    min_epochs=100,
    accelerator='gpu',
    gpus=2,
    num_nodes=1,
    log_every_n_steps=1,
    weights_save_path=os.environ['SHUBMAP_EXPS']
)

trainer.fit(
    model=model_holder,
    train_dataloaders=create_loader(train=False),
    val_dataloaders=create_loader(train=False),
)
