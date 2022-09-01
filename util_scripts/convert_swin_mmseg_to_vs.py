import os
import sys

import torch
from sys import argv
from typing import Dict

from hubmap_segmentation.models import create_model


if __name__ == '__main__':
    path_to_save = os.environ['PRETRAINED']
    path_to_mmseg_ckpt = argv[1]

    model_cfg = {
        'type': 'upernet',
        'backbone_cfg': {
            'type': 'swin',
            'use_norm': True
        }
    }

    our_model = create_model(model_cfg)

    our_st: Dict[str, torch.Tensor] = our_model.state_dict()
    ckpt_mmseg = torch.load(path_to_mmseg_ckpt, map_location='cpu')['state_dict']

    downsample_idx = 0

    st_dct = type(ckpt_mmseg)()
    for k, v in ckpt_mmseg.items():
        if 'seg' in k:
            print(k, v.shape)
            continue

        if 'backbone' not in k:
            st_dct[k] = v
            continue
        #k = k.replace('backbone.', '')
        if 'patch_embed' in k:
            k = k.replace('patch_embed', 'input_conv')
            k = k.replace('projection', '0')
            k = k.replace('norm', '2')
            st_dct[k] = v
            continue

        if 'downsample' in k:
            k = k.replace(f'stages.{downsample_idx}', f'merge_{downsample_idx}')
            k = k.replace('downsample.', '')
            st_dct[k] = v
            if 'bias' in k:
                downsample_idx += 1
            continue

        k = k.replace('stages.', 'layer_')
        k = k.replace('blocks.', '')
        k = k.replace('w_msa.', '')

        if 'ffn' in k:
            patt = 'ffn.layers.'
            suf = '0'
            if 'ffn.layers.0.0' in k:
                patt += '0.0'
            else:
                patt += '1'
                suf = '3'
            if patt not in k:
                print(patt, k)
            k = k.replace(patt, 'mlp.'+suf)
            #print(k)
            st_dct[k] = v
            continue

        if 'relative_position_index' in k:
            assert torch.all(v.view(-1) == our_st[k])
            v = v.view(-1).contiguous()

        st_dct[k] = v

    our_model.load_state_dict(st_dct, strict=False)
    # check if all is ok

    torch.save(
        our_model.state_dict(),
        os.path.join(path_to_save, 'swin_vs_small_ade20k.pth')
    )
