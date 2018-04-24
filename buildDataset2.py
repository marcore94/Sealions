
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import skimage.feature
import pandas as pd
import random
import cv2
import numpy as np
from joblib import Parallel, delayed


# In[2]:


SL_TRAIN_PATH = './data_set/train/0_sea_lions/'
SL_VALIDATION_PATH = './data_set/validation/0_sea_lions/'
SL_TEST_PATH = './data_set/test/0_sea_lions/'

BKG_TRAIN_PATH = './data_set/train/1_background/'
BKG_VALIDATION_PATH = './data_set/validation/1_background/'
BKG_TEST_PATH = './data_set/test/1_background/'


# In[3]:


# Read coordinates from files
sea_lions_df_train = pd.read_csv('./sealions_train.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})
sea_lions_df_validation = pd.read_csv('./sealions_validation.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})
sea_lions_df_test = pd.read_csv('./sealions_test.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})
empty_df_train = pd.read_csv('./empty_train.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})
empty_df_validation = pd.read_csv('./empty_validation.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})
empty_df_test = pd.read_csv('./empty_test.csv', dtype={"coord_x": int, "coord_y": int, "class": str, "filename": str})

# Use a random seed
random.seed(42)


# In[4]:


# Save 144x144 patches for sea lions
def save_sea_lion_patch_ext(t):
    image, coord_x, coord_y, path, number = t
    if coord_x < 72 or (coord_x > len(image[0]) - 72):
        return
    else:
        coord_x = coord_x - 72
    if coord_y < 72 or (coord_y > len(image) - 72):
        return
    else:
        coord_y = coord_y - 72
    patch = image[coord_y:coord_y+144, coord_x:coord_x+144, :]
    r = np.average(patch[:, :, 0])
    g = np.average(patch[:, :, 1])
    b = np.average(patch[:, :, 2])
    if b >= r and b >= g:
        return
    cv2.imwrite(path + str(number) + '.jpg', patch)


# Save 96x96 patches for sea lions
def save_sea_lion_patch(t):
    image, coord_x, coord_y, path, number = t
    if coord_x < 48 or (coord_x > len(image[0]) - 48):
        return
    else:
        coord_x = coord_x - 48
    if coord_y < 48 or (coord_y > len(image) - 48):
        coord_y = 0
    else:
        coord_y = coord_y - 48
    patch = image[coord_y:coord_y+96, coord_x:coord_x+96, :]
    r = np.average(patch[:, :, 0])
    g = np.average(patch[:, :, 1])
    b = np.average(patch[:, :, 2])
    if b >= r and b >= g:
        return
    cv2.imwrite(path + str(number) + '.jpg', patch)


# Save 96x96 patches for background
def save_background_patch(t):
    image, coord_x, coord_y, path, number = t
    patch = image[coord_y-48:coord_y+48, coord_x-48:coord_x+48, :]
    black = np.reshape(np.sum(patch, axis=2), (96*96))
    count_black = 0
    for px in black:
        if px == 0:
            count_black += 1
    if count_black >= int(0.02 * 96 * 96):
        return
    cv2.imwrite(path + str(number) + '.jpg', patch)


# In[5]:


def gen(img, path, lst):
    for l in lst:
        row = l[0]
        n = l[1]
        yield img, row[1]['coord_x'], row[1]['coord_y'], path, n


def extract_sea_lions_ext(sl_df, path):
    i = 0
    file_names = sl_df.filename.unique()
    for file in file_names:
        image = cv2.imread("../../home/shared/kaggle_sea_lions/data/Train/" + file)
        df = sl_df[(sl_df['filename'] == file) & (sl_df['class'] != "pup")]
        Parallel(n_jobs=4, verbose=0, backend="threading")(map(delayed(save_sea_lion_patch_ext), list(gen(image, path, list(zip(df.iterrows(), range(i, i+len(df))))))))
        i += len(df)


def extract_sea_lions(sl_df, path):
    i = 0
    file_names = sl_df.filename.unique()
    for file in file_names:
        image = cv2.imread("../../home/shared/kaggle_sea_lions/data/Train/" + file)
        df = sl_df[(sl_df['filename'] == file) & (sl_df['class'] != "pup")]
        Parallel(n_jobs=4, verbose=0, backend="threading")(map(delayed(save_sea_lion_patch), list(gen(image, path, list(zip(df.iterrows(), range(i, i+len(df))))))))
        i += len(df)


def extract_background(bkg_df, path):
    i = 0
    file_names = bkg_df.filename.unique()
    for file in file_names:
        image = cv2.imread("../../home/shared/kaggle_sea_lions/data/TrainDotted/" + file)
        df = bkg_df[bkg_df['filename'] == file]
        Parallel(n_jobs=4, verbose=0, backend="threading")(map(delayed(save_background_patch), list(gen(image, path, list(zip(df.iterrows(), range(i, i+len(df))))))))
        i += len(df)


# In[6]:


# Extract background train patches
extract_background(empty_df_train, BKG_TRAIN_PATH)

# Extract background validation patches
extract_background(empty_df_validation, BKG_VALIDATION_PATH)

# extract background test patches
extract_background(empty_df_test, BKG_TEST_PATH)


# In[7]:


# Extract sea lions train patches
extract_sea_lions(sea_lions_df_train, SL_TRAIN_PATH)


# In[6]:


# Extract sea lions validation patches
extract_sea_lions(sea_lions_df_validation, SL_VALIDATION_PATH)

# extract sea lions test patches
extract_sea_lions(sea_lions_df_test, SL_TEST_PATH)
