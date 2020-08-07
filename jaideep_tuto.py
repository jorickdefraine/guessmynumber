# Handwritten Digit Recognition GUI App by jaideep singh
# 1. Import the libraries and load the MNIST dataset

# import libraries

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import to_categorical
from keras import backend as K
import numpy
import pandas as pd

# load dataset directly from keras library
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot first six samples of MNIST training dataset as gray scale image
import matplotlib.pyplot as plt

for i in range(6):
    plt.subplot(int('23' + str(i + 1)))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

# 2. Data Preprocess and Normalize

# reshape format [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Converts a class vector (integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# normalise inputs
X_train = X_train / 255
X_test = X_test / 255


# 3. Create the model

# define a CNN model
def create_model():
    num_classes = 10
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# buld the model
model = create_model()

# 4. Train the model

#fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=200, verbose=2)
print("The model has successfully trained")

#Save the model
model.save('model_jaideep.h5')
print("The model has successfully saved")

# 5. Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# 6. Create GUI to predict digits
