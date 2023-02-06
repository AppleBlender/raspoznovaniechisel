import bible
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


def _mnist_cnn_model():
   image_size = 28
   num_channels = 1  # 1 for grayscale images
   num_classes = 10  # Number of outputs
   model = Sequential()
   model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu',
            padding='same',
   input_shape=(image_size, image_size, num_channels)))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
            padding='same'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Flatten())
   # Densely connected layers
   model.add(Dense(128, activation='relu'))
   # Output layer
   model.add(Dense(num_classes, activation='softmax'))
   model.compile(optimizer=Adam(), loss='categorical_crossentropy',
            metrics=['accuracy'])
   return model