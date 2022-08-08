import os
import json

import numpy as np
import pandas as pd

from tqdm.auto import trange
from sys import argv
from PIL import Image


def mega_crop(
        tensor: np.array,
        mask: np.array,
        h: int,
        id: str,
        folder_path: str,
        folder_mask_path: str,
        H: int,
        idx: int = 0
    ) -> int:

    for h_l in range(0, h - 1, H):
        h_r = h_l + H
        for w_l in range(0, h - 1, H):
            w_r = w_l + H
            sub_tensor = tensor[h_l:h_r, w_l:w_r]
            sub_mask = mask[h_l:h_r, w_l:w_r]

            #print(mask.shape, h_l, h_r, h_r - h_l, sub_mask.shape, flush=True)
            if np.std(sub_tensor) < 5:
                continue

            image_path = os.path.join(folder_path, id + '_{}'.format(idx))
            np.save(image_path, sub_tensor)

            mask_path = os.path.join(folder_mask_path, id + '_{}'.format(idx))
            np.save(mask_path, sub_mask)

            idx += 1

    return idx


def tile_masks_images():
    if 'SDATASET_PATH' in os.environ:
        root = os.environ['SDATASET_PATH']
        H = int(argv[1])
    else:
        root = argv[1]
        H = int(argv[2])

    csv_path = os.path.join(root, 'train.csv')
    df = pd.read_csv(csv_path)

    new_masks_path = os.path.join(root, 'tiled_images/masks_{}'.format(H))
    new_images_path = os.path.join(root, 'tiled_images/images_{}'.format(H))

    if not os.path.exists(new_masks_path):
        os.makedirs(new_masks_path)
    if not os.path.exists(new_images_path):
        os.makedirs(new_images_path)


    masks_full_ref = os.path.join(root, 'full_masks')
    images_full_ref = os.path.join(root, 'full_images')

    for idx in trange(len(df)):
        row = df.iloc[idx]
        h = row['img_height']
        w = row['img_width']

        id = row['id']
        id = str(id)

        mask = np.load(os.path.join(masks_full_ref, id + '.npy'))
        image = np.load(os.path.join(images_full_ref, id + '.npy'))

        mask = Image.fromarray(mask)
        image = Image.fromarray(image)

        new_h = ((h - 1) // H + 1) * H # usually 3072 for h=3000 and H=512

        mask = mask.resize((new_h, new_h), resample=Image.LANCZOS)
        image = image.resize((new_h, new_h), resample=Image.LANCZOS)

        mask = np.array(mask, dtype=np.float32)
        image = np.array(image, dtype=np.uint8)

        idx_image = mega_crop(image, mask, new_h, id, new_images_path, new_masks_path, H, 0)

        shift = H // 2

        image = image[shift:-shift, shift:-shift]
        mask = mask[shift:-shift, shift:-shift]

        idx_image = mega_crop(image, mask, new_h - H, id, new_images_path, new_masks_path, H, idx_image)



if __name__ == '__main__':
    tile_masks_images()
