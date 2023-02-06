import bible
import mnist_cnn_model as md
import mnist_cnn_train
from tensorflow import keras
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
#from tensorflow.keras.optimizers import SGD, RMSprop, Adam

model = md._mnist_cnn_model()
mnist_cnn_train._mnist_cnn_train(model)
model.save('cnn_digits_28x28.h5')