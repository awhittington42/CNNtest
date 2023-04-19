import sys
import sklearn
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt


print("Preparing datasets:")
#load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#divide by 255 to normalize
X_train, X_test = X_train / 255.0, X_test / 255.0
print("classes:")
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
print('encodings:')
#prepare y variables for comparison to categorical values (0-9)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
stat = ("Classes: " + str(num_classes) + ", y_train: " + str(len(y_train)) + ", y_test: " + str(len(y_test)) + ", X_train: " + str(len(X_train)) + ", X_test: " + str(len(X_test)))
print(stat)
print("Checking for images")
plt.subplot(2, 2, 1)
plt.imshow(X_train[7], cmap=plt.get_cmap('CMRmap'))
plt.subplot(2, 2, 2)
plt.imshow(X_train[77], cmap=plt.get_cmap('CMRmap'))
plt.subplot(2, 2, 3)
plt.imshow(X_train[777], cmap=plt.get_cmap('CMRmap'))
plt.subplot(2, 2, 4)
plt.imshow(X_train[7777], cmap=plt.get_cmap('CMRmap'))
plt.show()
print("Data prepared")

print("Making validation sets")
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]

#Method 1: Basic CNN

def basic_cnn():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
  model.add(MaxPooling2D(strides=(2, 2)))
  model.add(Conv2D(64, (3, 3), activation = 'relu'))
  model.add(MaxPooling2D(strides = (2, 2)))
  model.add(Conv2D(32, (3, 3), activation = 'relu'))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(num_classes, activation = 'softmax'))
  model.compile(loss='mean_squared_error', optimizer = 'adam', metrics=['accuracy'])
  return model

basic_model = basic_cnn()
print(basic_model.summary())
print("Fitting")
results = basic_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size= 50)
scores = basic_model.evaluate(X_test, y_test, verbose=0)
print("Basic CNN accuracy: %.2f%%" % (scores[1]*100))
