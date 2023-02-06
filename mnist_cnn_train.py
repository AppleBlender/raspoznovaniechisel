import mnist_cnn_model
import bible
#from tensorflow.keras import Sequential
from tensorflow import keras
import numpy as np
import time
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
#from tensorflow.keras.optimizers import SGD, RMSprop, Adam


def _mnist_cnn_train(model):
   (train_digits, train_labels), (test_digits, test_labels) = keras.datasets.mnist.load_data()

   # Get image size
   image_size = 28
   num_channels = 1  # 1 for grayscale images

   # re-shape and re-scale the images data
   train_data = np.reshape(train_digits, (train_digits.shape[0], image_size, image_size, num_channels))
   train_data = train_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
   num_classes = 10
   train_labels_cat = keras.utils.to_categorical(train_labels, num_classes)

   # re-shape and re-scale the images validation data
   val_data = np.reshape(test_digits, (test_digits.shape[0], image_size, image_size, num_channels))
   val_data = val_data.astype('float32') / 255.0
   # encode the labels - we have 10 output classes
   val_labels_cat = keras.utils.to_categorical(test_labels, num_classes)

   print("Training the network...")
   t_start = time.time()

   # Start training the network
   model.fit(train_data, train_labels_cat, epochs=8, batch_size=64,
        validation_data=(val_data, val_labels_cat))

   print("Done, dT:", time.time() - t_start)

   return model