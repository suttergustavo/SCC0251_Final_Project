"""
Given all the images in one single folder (the one that came on the zip file)
separetes them in two folders, train (80% of the images) and val (20% of the images)
"""

import os
import numpy as np


n_files = 202599
original_folder = '../img_align_celeba'
train_folder = '../train'
val_folder = '../val'


all_files = np.arange(1, n_files + 1)
np.random.shuffle(all_files)

train = all_files[:int(0.8 * n_files)]
val = all_files[int(0.8 * n_files):]

for i, file_idx in enumerate(train):
    os.rename(f'{original_folder}/{file_idx:06d}.jpg', f'{train_folder}/{i:06d}.jpg')

for i, file_idx in enumerate(val):
    os.rename(f'{original_folder}/{file_idx:06d}.jpg', f'{val_folder}/{i:06d}.jpg')