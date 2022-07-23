import os

import numpy as np
import pandas as pd

from sys import argv
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
    else:
        root = argv[1]

    random_seed = 1_019_541

    df = pd.read_csv(os.path.join(root, 'train.csv'))

    train_df = pd.DataFrame(columns=df.columns)
    valid_df = pd.DataFrame(columns=df.columns)

    for organ, subdf in df.groupby(['organ']):
        sub_train_df, sub_valid_df = train_test_split(
            subdf,
            test_size=0.2,
            random_state=random_seed,
            shuffle=True,
        )
        train_df = pd.concat([train_df, sub_train_df])
        valid_df = pd.concat([valid_df, sub_valid_df])

    train_df.to_csv(os.path.join(root, 'csv_files', 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(root, 'csv_files', 'valid.csv'), index=False)

