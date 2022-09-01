from .holder import ModelHolder
from .freeze_holder import FreezeHolder


def create_holder(holder_cfg):
    if 'type' not in holder_cfg:
        holder_cfg['type'] = 'base'
    type = holder_cfg.pop('type')
    if type == 'base':
        return ModelHolder(**holder_cfg)
    return FreezeHolder(**holder_cfg)