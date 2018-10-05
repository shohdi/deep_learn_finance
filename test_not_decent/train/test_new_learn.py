from __future__ import division , print_function
from keras.models import Sequential
from keras.layers.core import Activation,Dense , Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from scipy.misc import imresize
import collections
import numpy as np
import os



model = Sequential()
#model.add(Conv2D(32,kernel_size=8,strides=4,kernel_initializer="normal",padding="same",input_shape=(80,80,4)))
#model.add(Activation("relu"))
#model.add(Conv2D(64,kernel_size=4,strides=2,kernel_initializer="normal",padding="same"))
#model.add(Activation("relu"))
#model.add(Conv2D(64,kernel_size=3,strides=1,kernel_initializer="normal",padding="same"))
#model.add(Activation("relu"))
#model.add(Flatten())

model.add(Dense(512,kernel_initializer="normal",input_shape=(80*80*4,)))
model.add(Activation("relu"))
model.add(Dense(3,kernel_initializer="normal"))
model.compile(optimizer=Adam(lr=1e-6),loss="mse")


model.summary();

oldWeights = model.get_weights()

arr = np.array(oldWeights);

print(arr.shape)

arr = arr.flatten().flatten().flatten();
print(arr.shape);
print('test1');



