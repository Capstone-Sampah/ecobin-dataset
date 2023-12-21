import os
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.layers.experimental import preprocessing
from keras.utils import plot_model

physical_devices = tf.config.list_physical_devices()
if len(physical_devices) == 0:
    print("No devices found.")
else:
    for device in physical_devices:
        print(f"Device name: {device.name}, type: {device.device_type}")

PATH_DATASET=r"mnt/c/Users/Felicia Pangestu/Documents/BANGKIT/Capstone/splitedDataset"
TRAIN_DIR=r"/mnt/c/Users/Felicia Pangestu/Documents/BANGKIT/Capstone/splitedDataset/train_ds"
VAL_DIR=r"/mnt/c/Users/Felicia Pangestu/Documents/BANGKIT/Capstone/splitedDataset/val_ds"
TEST_DIR=r"/mnt/c/Users/Felicia Pangestu/Documents/BANGKIT/Capstone/splitedDataset/test_ds"

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]

def create_datasets(path_dataset):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        seed=0,
        batch_size=20,
        label_mode='categorical',
        image_size=(256, 256),
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        seed=0,
        batch_size=20,
        label_mode='categorical',
        image_size=(256, 256),
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        seed=0,
        batch_size=20,
        label_mode='categorical',
        image_size=(256, 256),
    )

    return train_ds, validation_ds, test_ds

# Create datasets with or without data augmentation
train_ds, validation_ds, test_ds = create_datasets(PATH_DATASET)

# https://keras.io/api/applications/efficientnet_v2/#efficientnetv2s-function
IMG_SHAPE = (256, 256, 3)
base_model = keras.applications.EfficientNetV2S(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
    include_preprocessing=False,
)
# fine tuning - https://keras.io/guides/transfer_learning/#freezing-layers-understanding-the-trainable-attribute
base_model.trainable = False
# base_model.summary()

tuning_layer_name = 'block5a_expand_conv'
tuning_layer = base_model.get_layer(tuning_layer_name)
tuning_index = base_model.layers.index(tuning_layer)
for layer in base_model.layers[:tuning_index]:
    layer.trainable =  False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.experimental.preprocessing.RandomContrast(0.3),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
], name='data_augmentation')

model = Sequential([
    data_augmentation,
    base_model,
    Conv2D(256, (3, 3), activation='relu', input_shape=IMG_SHAPE),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

learning_rate = 0.0001
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)

epochs = 50
batch_size = 48
history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, batch_size=batch_size)

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
history = model.fit(train_ds, epochs=epochs, validation_data=validation_ds, batch_size=batch_size)

MODEL_PATH = r"/mnt/c/Users/Felicia Pangestu/Documents/BANGKIT/Capstone/model/ecobin_model8.h5"
model.save(MODEL_PATH)