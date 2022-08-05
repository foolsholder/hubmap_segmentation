import os

import numpy as np
import pandas as pd

from sys import argv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
