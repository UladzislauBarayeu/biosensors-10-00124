from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Flatten, Reshape, BatchNormalization, Activation, Conv2DTranspose, AveragePooling2D
from keras.layers import Dropout, ZeroPadding2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, Dense, UpSampling3D
import os
import sys
sys.path.insert(0, '../')
from configurations import *
from network_utils import *


input_shape = Input(shape=(18, 64, 1))
dense_dim=36

x = Conv2D(filters=4, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(input_shape)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=4, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=4, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x=MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)

x = Conv2D(filters=8, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x = Conv2D(filters=8, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=8, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x=MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)

x = Conv2D(filters=16, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=16, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x=MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)

x = Conv2D(filters=32, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x=Dropout(rate=0.2)(x)

x = Conv2D(filters=32, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x=MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)

x = Conv2D(filters=64, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x=MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="same")(x)

x = Conv2D(filters=32, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=32, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x = Conv2D(filters=16, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x = Conv2D(filters=16, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=4, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same', bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=4, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x = Conv2D(filters=1, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)


x=Flatten()(x)

x=Dense(dense_dim, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x=Dense(dense_dim, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)

x=Dropout(rate=0.2)(x)

x=Dense(dense_dim, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x=Dense(dense_dim, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)

x=Dense(dense_dim//2, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)
x=Dense(dense_dim//2, activation="relu",  bias_initializer="glorot_normal", kernel_initializer="glorot_normal")(x)

output = Dense(2, activation='softmax')(x)

network = Model(input_shape, output, name="nn")
network.summary()

save_network("simple_1_with_dropout_3", network)