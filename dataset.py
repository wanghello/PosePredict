#################################################     
#     Filename: dataset.py
#       Author: wang - shawn_wang@163.com
#  Description: ---
#       Create: 2019-12-18 22:36:56
#Last Modified: 2019-12-18 22:36:56
#################################################     
#!/usr/bin/env python
# coding=utf-8
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from utils.util import *

PATH = "./data/"
train_images_dir = PATH + 'train_images/{}.jpg'
train = pd.read_csv(PATH + 'train.csv')

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(2) == 1

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=False)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]

def load_dataset(cross_validation, BATCH_SIZE):
    if cross_validation:
        df_train, df_dev = train_test_split(train, test_size=0.1)
        train_dataset = CarDataset(df_train, train_images_dir, training=True)
        dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
    else:
        df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
        train_dataset = CarDataset(df_train, train_images_dir, training=True)
        dev_dataset = CarDataset(df_dev, train_images_dir, training=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    return train_loader, dev_loader

def load_test_dataset(BATCH_SIZE):
    pass
