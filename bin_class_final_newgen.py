
# coding: utf-8

# In[1]:


import keras
import cv2
from skimage.transform import rotate, resize, SimilarityTransform, warp
import os
import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import copy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.callbacks import ModelCheckpoint


# In[2]:


INPUT_SHAPE = (96, 96, 3)
IM_HEIGHT = 96
IM_WIDTH = 96
OUTPUT_SIZE = 2

LEARNING_RATE = 0.0005
OPTIMIZER = keras.optimizers.Adam(lr=LEARNING_RATE)
LOSS = 'binary_crossentropy'
METRIC = 'accuracy'

SL_TRAIN_SIZE = 40411
SL_VALIDATION_SIZE = 9668
SL_TEST_SIZE = 13539
EPOCHS = 25
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 50
STEPS_PER_EPOCH = (2 * SL_TRAIN_SIZE) // BATCH_SIZE + 1
VALIDATION_STEPS_PER_EPOCH = (2 * SL_VALIDATION_SIZE) // BATCH_SIZE + 1
MAX_EPOCHS_WITH_SAME_DATA_SET = 5

TRAIN_PATH = "./data_set/train/"
VALIDATION_PATH = "./data_set/validation/"
TEST_PATH = "./data_set/test/"

MODEL_PATH = "./binary_classifier/net_2_model.h5"

MR_CKPT_PATH = "./binary_classifier/net_2_most_recent_checkpoint.hdf5"
CB_CKPT_PATH = "./binary_classifier/net_2_current_best_checkpoint.hdf5"


# In[3]:


train_datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
    rescale=1./255,)


# In[4]:


def train_generator():
    sl_lst_tmp = os.listdir(TRAIN_PATH + '0_sea_lions')
    bkg_lst_tmp = os.listdir(TRAIN_PATH + '1_background')
    for i in range(EPOCHS // MAX_EPOCHS_WITH_SAME_DATA_SET):
        Y_train = np.array([[1, 0]]*SL_TRAIN_SIZE + [[0, 1]]*SL_TRAIN_SIZE)
        img_lst = []
        for name in sl_lst_tmp:
            img = cv2.imread(TRAIN_PATH + '0_sea_lions/' + name)
            img_lst.append(img[72-48:72+48, 72-48:72+48, :])
        for name in rand.sample(bkg_lst_tmp, SL_TRAIN_SIZE):
            img_lst.append(cv2.imread(TRAIN_PATH + '1_background/' + name))
        X_train = np.array(img_lst)
        gen = train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
        for j in range(MAX_EPOCHS_WITH_SAME_DATA_SET):
            step = 0
            for batch in gen:
                yield batch
                step += 1
                if step >= STEPS_PER_EPOCH:
                    break


# In[5]:


# Validation data set
X_validation = []
Y_validation = []
validation_set = []
lst = os.listdir(VALIDATION_PATH + '0_sea_lions')
for elem in lst:
    validation_set.append(list((cv2.imread(VALIDATION_PATH + '0_sea_lions/' + elem), 'sea_lion')))
lst = os.listdir(VALIDATION_PATH + '1_background')
for elem in lst:
    validation_set.append(list((cv2.imread(VALIDATION_PATH + '1_background/' + elem), 'background')))
rand.shuffle(validation_set)
for data in validation_set:
    X_validation.append(data[0])
    if data[1] == 'sea_lion':
        Y_validation.append([1, 0])
    else:
        Y_validation.append([0, 1])

X_validation = np.array(X_validation, copy=False)
# Convert data types and normalize values
X_validation = X_validation.astype('float32')
X_validation /= 255
Y_validation = np.array(Y_validation, copy=False)

# Free memory
lst = []
validation_set = []


# In[6]:


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracies = []
        self.val_acc = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracies.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


history = History()


# In[7]:


# Build model

model = Sequential()
# First layer
model.add(Convolution2D(8, (5, 5), activation='relu', padding='valid', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second layer
model.add(Convolution2D(5, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third layer
model.add(Convolution2D(5, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth layer
model.add(Convolution2D(10, (3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(OUTPUT_SIZE, activation='softmax'))

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[METRIC])


# In[8]:


# Checkpointers

# most recent checkpoint
mr_checkpointer = ModelCheckpoint(filepath=MR_CKPT_PATH, verbose=1, save_best_only=False)
# current best checkpoint
cb_checkpointer = ModelCheckpoint(filepath=CB_CKPT_PATH, verbose=1, save_best_only=True)


# In[9]:


# Train

# Fit model on training data
results = model.fit_generator(
    train_generator(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_validation, Y_validation),
    workers=8,
    max_queue_size=50,
    callbacks=[mr_checkpointer, cb_checkpointer, history])


# In[ ]:


# Save history
h_df = pd.DataFrame({'acc': history.accuracies,
                     'val_acc': history.val_acc,
                     'loss': history.losses,
                     'val_loss': history.val_losses})

h_df.to_csv("./metrics.csv", index=False)


# In[9]:


# Save trained model

# serialize model to HDF5
model.save(MODEL_PATH)
