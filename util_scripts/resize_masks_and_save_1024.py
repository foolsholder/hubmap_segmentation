import os
import json

import numpy as np
import pandas as pd

from tqdm.auto import trange
from sys import argv
from PIL import Image


def create_masks():
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
    else:
        root = argv[1]

    H = 1024

    koef = 1024 / 3000.

    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    new_masks_path = os.path.join(root, 'resized_images/masks_scaled_1024')
    if not os.path.exists(new_masks_path):
        os.makedirs(new_masks_path)

    for idx in trange(len(df)):
        row = df.iloc[idx]
        h = row['img_height']
        w = row['img_width']

        new_h = int(koef * h + 1e-2)
        new_w = int(koef * w + 1e-2)

        tmp = np.zeros((h * w), dtype=np.float32)
        id = row['id']
        id = str(id)

        arr = row['rle'].split()
        arr = np.array(arr, dtype=np.int32)
        fst, snd = arr[0::2] - 1, arr[1::2]
        list_of_pairs = zip(fst, snd)
        for (i, length) in list_of_pairs:
            i -= 1
            tmp[i:i+length] = 1

        tmp = tmp.reshape((w, h)).T

        tmp = Image.fromarray(tmp)
        tmp = tmp.resize(size=(new_h, new_w), resample=Image.LANCZOS)
        tmp = np.array(tmp, dtype=np.float32)

        np.save(os.path.join(new_masks_path, id), tmp)


def create_images():
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
    else:
        root = argv[1]

    H = 1024

    koef = 1024 / 3000.

    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    new_images_path = os.path.join(root, 'resized_images/images_scaled_1024')

    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)

    for idx in trange(len(df)):
        row = df.iloc[idx]
        h = row['img_height']
        w = row['img_width']
        id = row['id']
        id = str(id)

        new_h = int(koef * h + 1e-2)
        new_w = int(koef * w + 1e-2)

        tmp = Image.open(os.path.join(root, 'train_images', id + '.tiff'))

        tmp = tmp.resize(size=(new_h, new_w), resample=Image.LANCZOS)
        tmp = np.array(tmp)
        np.save(os.path.join(new_images_path, id), tmp)


if __name__ == '__main__':
    create_masks()
    create_images()
