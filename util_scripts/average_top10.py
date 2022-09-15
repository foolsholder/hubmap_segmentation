import os
import json

import numpy as np
import pandas as pd
import torch

from glob import glob
from sys import argv

if __name__ == '__main__':
    path = argv[1]

    list_ckpt = glob(path + '/epoch=*.ckpt')
    print(list_ckpt)

    st = torch.load(list_ckpt[0], map_location='cpu')['state_dict']

    for idx in range(1, len(list_ckpt)):
        st_new = torch.load(list_ckpt[idx], map_location='cpu')['state_dict']
        for k, v in st_new.items():
            st[k] += v

    for k, v in st.items():
        st[k] = v / len(list_ckpt)

    dct = type(st)()
    dct['state_dict'] = st
    torch.save(dct, path + '/avg_top10.ckpt')