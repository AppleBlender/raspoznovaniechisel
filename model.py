import bible
import mnist_make_model as md
import mnist_mlp_train

model = md._mnist_make_model(image_w=28, image_h=28)
mnist_mlp_train._mnist_mlp_train(model)
model.save('mlp_digits_28x28.h5')