import os
import json

import numpy as np
import pandas as pd

from tqdm.auto import trange
from sys import argv
from PIL import Image


def pad(tmp):
    pad_h = 3000 - tmp.shape[0]
    pad_w = 3000 - tmp.shape[1]

    if pad_w < 0 or pad_h < 0:
        return tmp

    fst_h = pad_h // 2
    snd_h = pad_h - fst_h

    fst_w = pad_w // 2
    snd_w = pad_w - fst_w

    if len(tmp.shape) == 3:
        return np.pad(tmp, ((fst_h, snd_h), (fst_w,snd_w), (0, 0)), constant_values=0)
    return np.pad(tmp, ((fst_h, snd_h), (fst_w,snd_w)), constant_values=0)

def create_masks():
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
        H = int(argv[1])
    else:
        root = argv[1]
        H = int(argv[2])

    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    new_masks_path = os.path.join(root, 'resized_images/masks_{}'.format(H))
    if not os.path.exists(new_masks_path):
        os.makedirs(new_masks_path)

    masks_full_ref = os.path.join(root, 'full_masks')
    flag = True
    if not os.path.exists(masks_full_ref):
        os.makedirs(masks_full_ref)
        flag = True

    for idx in trange(len(df)):
        row = df.iloc[idx]
        h = row['img_height']
        w = row['img_width']

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
        tmp = pad(tmp)


        if flag:
            np.save(os.path.join(masks_full_ref, id), tmp)
        tmp = Image.fromarray(tmp)
        tmp = tmp.resize(size=(H, H), resample=Image.LANCZOS)
        tmp = np.array(tmp, dtype=np.float32)

        np.save(os.path.join(new_masks_path, id), tmp)


def create_images():
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
        H = int(argv[1])
    else:
        root = argv[1]
        H = int(argv[2])

    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    new_images_path = os.path.join(root, 'resized_images/images_{}'.format(H))

    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)

    images_full_ref = os.path.join(root, 'full_images')
    flag = True
    if not os.path.exists(images_full_ref):
        os.makedirs(images_full_ref)
        flag = True

    for idx in trange(len(df)):
        row = df.iloc[idx]
        id = row['id']
        id = str(id)

        tmp = Image.open(os.path.join(root, 'train_images', id + '.tiff'))
        tmp = np.array(tmp)
        tmp = pad(tmp)
        if flag:
            np.save(os.path.join(images_full_ref, id), tmp)
        tmp = Image.fromarray(tmp)
        tmp = tmp.resize(size=(H, H), resample=Image.LANCZOS)
        tmp = np.array(tmp)

        np.save(os.path.join(new_images_path, id), tmp)


if __name__ == '__main__':
    create_masks()
    #create_images()
