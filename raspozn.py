import bible
import mnist_cnn_model
import mnist_cnn_train
from tensorflow import keras
import tensorflow as tf
import numpy as np

def cnn_digits_predict(model, image_file):
   image_size = 28
   img = keras.preprocessing.image.load_img(image_file,
target_size=(image_size, image_size), color_mode='grayscale')
   img_arr = np.expand_dims(img, axis=0)
   img_arr = 1 - img_arr/255.0
   img_arr = img_arr.reshape((1, 28, 28, 1))

   result = model.predict([img_arr])
   return result[0]

model = tf.keras.models.load_model('cnn_digits_28x28.h5')
print(cnn_digits_predict(model, 'C:\\Users\\anxie\\Desktop\\kursach\\venv\\Lib\\site-packages\\keras\\digit_0.png'))
print(cnn_digits_predict(model, 'C:\\Users\\anxie\\Desktop\\kursach\\venv\\Lib\\site-packages\\keras\\digit_1.png'))
print(cnn_digits_predict(model, 'C:\\Users\\anxie\\Desktop\\kursach\\venv\\Lib\\site-packages\\keras\\digit_3.png'))
print(cnn_digits_predict(model, 'C:\\Users\\anxie\\Desktop\\kursach\\venv\\Lib\\site-packages\\keras\\digit_8.png'))
print(cnn_digits_predict(model, 'C:\\Users\\anxie\\Desktop\\kursach\\venv\\Lib\\site-packages\\keras\\digit_9.png'))