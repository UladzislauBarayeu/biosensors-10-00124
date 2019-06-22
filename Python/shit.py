import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np
import os
import random

from keras.utils import plot_model

input_shape = Input(shape=(176, 200, 3))
x=Conv2D(32, (3, 3), padding='same',
                 input_shape=(176, 200, 3),
                 activation='relu')(input_shape)
x=Conv2D(32, (3, 3), activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.2)(x)

x=Conv2D(64, (3, 3), padding='same',
                 activation='relu')(x)
x=Conv2D(64, (3, 3), activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.2)(x)

x=Conv2D(128, (3, 3), padding='same',
                 activation='relu')(x)
x=Conv2D(128, (3, 3), activation='relu')(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
x=Dropout(0.2)(x)
x=Flatten()(x)
x=Dense(512, activation='relu')(x)
x=Dropout(0.5)(x)
x=Dense(4, activation='softmax')(x)
model = Model(input_shape, x, name="nn")
model.summary()

plot_model(model, to_file='model.png', show_layer_names=False, show_shapes =True, rankdir='LR' )