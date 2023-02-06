import bible
from tensorflow.python import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_datasets as tfds
import tensorflow as tf
import logging
import numpy as np

def _mnist_make_model(image_w: int, image_h: int):
   # Neural network model
   model = Sequential()
   model.add(Dense(784, activation='relu', input_shape=(image_w*image_h,)))
   model.add(Dense(10, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
   return model