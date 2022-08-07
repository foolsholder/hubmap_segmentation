import os

import numpy as np
import pandas as pd

from glob import glob
from copy import copy
from tqdm import tqdm, trange
from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def create_tiled_and_filter(df, H):
    new_df = pd.DataFrame(columns=df.columns)

    for idx in trange(len(df)):
        row = df.iloc[idx]

        id = str(row['id'])

        new_masks_path = os.path.join(root, 'tiled_images/masks_{}'.format(H))
        new_images_path = os.path.join(root, 'tiled_images/images_{}'.format(H))

        image_patt = os.path.join(new_images_path, id + '_*')
        masks_patt = os.path.join(new_masks_path, id + '_*')

        tiles_img = glob(image_patt)
        tiles_msk = glob(masks_patt)

        for img_path, mask_path in zip(tiles_img, tiles_msk):
            # .npy - 4
            new_id = img_path.split('/')[-1][:-4]
            row = copy(df.iloc[idx])
            row['id'] = new_id
            subdf = row.to_frame().T
            new_df = pd.concat([new_df, subdf])
    return new_df


if __name__ == '__main__':
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
    else:
        root = argv[1]

    random_seed = 1_019_541

    df = pd.read_csv(os.path.join(root, 'csv_files', 'train.csv'))

    K = 4
    splitter = KFold(n_splits=K, shuffle=True, random_state=random_seed)

    train_dfs = []
    valid_dfs = []
    for i in range(K):
        train_dfs += [pd.DataFrame(columns=df.columns)]
        valid_dfs += [pd.DataFrame(columns=df.columns)]

    for organ, subdf in df.groupby(['organ']):
        idx = 0
        for train_idx, valid_idx in splitter.split(subdf):
            train_dfs[idx] = pd.concat([subdf.iloc[train_idx], train_dfs[idx]])
            valid_dfs[idx] = pd.concat([subdf.iloc[valid_idx], valid_dfs[idx]])
            idx += 1

    for idx, (train_df, valid_df) in enumerate(zip(train_dfs, valid_dfs)):
        train_df.to_csv(
            os.path.join(root, 'csv_files', 'train_{}.csv'.format(idx)),
            index=False)
        valid_df.to_csv(
            os.path.join(root, 'csv_files', 'valid_{}.csv'.format(idx)),
            index=False)

        tiled_train_df = create_tiled_and_filter(train_df, H=1024)
        tiled_valid_df = create_tiled_and_filter(valid_df, H=512)

        tiled_train_df.to_csv(
            os.path.join(root, 'csv_files', 'tiled_train_{}.csv'.format(idx)),
            index=False)
        tiled_valid_df.to_csv(
            os.path.join(root, 'csv_files', 'tiled_valid_{}.csv'.format(idx)),
            index=False)
