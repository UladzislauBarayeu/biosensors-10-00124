import keras
import tensorflow as tf
from keras.layers import LeakyReLU, Dense
from keras.models import load_model
from keras import optimizers
import numpy as np
from keras.models import model_from_json
from keras.layers import Input
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Dense, Flatten, Reshape, BatchNormalization, Activation, Conv2DTranspose, AveragePooling2D
from keras.layers import Dropout, ZeroPadding2D, Cropping2D, Concatenate
from keras.layers import LeakyReLU, Dense, UpSampling3D
import os
from configurations import *

def make_nn(nn, two_task=True):
    input_shape = Input(shape=(40, 64, 1))
    encoding_dim = 80
    conv2d_1 = Conv2D(filters=32, kernel_size=(1, 3), activation='linear', strides=(1, 2), padding='same',
                      name='conv2d_1')(input_shape)
    batch_normalization_1 = BatchNormalization(name='batch_normalization_1')(conv2d_1)
    activation_1 = Activation('relu', name='activation_1')(batch_normalization_1)

    conv2d_2 = Conv2D(filters=32, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_2')(activation_1)
    batch_normalization_2 = BatchNormalization(name='batch_normalization_2')(conv2d_2)
    activation_2 = Activation('relu', name='activation_2')(batch_normalization_2)

    conv2d_3 = Conv2D(filters=64, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_3')(activation_2)
    batch_normalization_3 = BatchNormalization(name='batch_normalization_3')(conv2d_3)
    activation_3 = Activation('relu', name='activation_3')(batch_normalization_3)

    max_pooling2d_1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='valid', name='max_pooling2d_1')(
        activation_3)

    conv2d_4 = Conv2D(filters=80, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_4')(max_pooling2d_1)
    batch_normalization_4 = BatchNormalization(name='batch_normalization_4')(conv2d_4)
    activation_4 = Activation('relu', name='activation_4')(batch_normalization_4)

    conv2d_5 = Conv2D(filters=192, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_5')(activation_4)
    batch_normalization_5 = BatchNormalization(name='batch_normalization_5')(conv2d_5)
    activation_5 = Activation('relu', name='activation_5')(batch_normalization_5)

    max_pooling2d_2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='valid', name='max_pooling2d_2')(
        activation_5)

    ##

    conv2d_9 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_9')(max_pooling2d_2)
    batch_normalization_9 = BatchNormalization(name='batch_normalization_9')(conv2d_9)
    activation_9 = Activation('relu', name='activation_9')(batch_normalization_9)

    conv2d_7 = Conv2D(filters=48, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_7')(max_pooling2d_2)
    conv2d_10 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_10')(activation_9)
    batch_normalization_7 = BatchNormalization(name='batch_normalization_7')(conv2d_7)
    batch_normalization_10 = BatchNormalization(name='batch_normalization_10')(conv2d_10)
    activation_7 = Activation('relu', name='activation_7')(batch_normalization_7)
    activation_10 = Activation('relu', name='activation_10')(batch_normalization_10)

    # average_pooling2d_1 = AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding='same',
    #                                        name='average_pooling2d_1')(max_pooling2d_2)

    conv2d_6 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_6')(max_pooling2d_2)
    conv2d_8 = Conv2D(filters=64, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same',
                      name='conv2d_8')(activation_7)
    conv2d_11 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_11')(activation_10)
    conv2d_12 = Conv2D(filters=32, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_12')(max_pooling2d_2)
    batch_normalization_6 = BatchNormalization(name='batch_normalization_6')(conv2d_6)
    batch_normalization_8 = BatchNormalization(name='batch_normalization_8')(conv2d_8)
    batch_normalization_11 = BatchNormalization(name='batch_normalization_11')(conv2d_11)
    batch_normalization_12 = BatchNormalization(name='batch_normalization_12')(conv2d_12)
    activation_6 = Activation('relu', name='activation_6')(batch_normalization_6)
    activation_8 = Activation('relu', name='activation_8')(batch_normalization_8)
    activation_11 = Activation('relu', name='activation_11')(batch_normalization_11)
    activation_12 = Activation('relu', name='activation_12')(batch_normalization_12)

    mixed0 = Concatenate(name='mixed0')([activation_6, activation_8, activation_11, activation_12])

    conv2d_16 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_16')(mixed0)
    batch_normalization_16 = BatchNormalization(name='batch_normalization_16')(conv2d_16)
    activation_16 = Activation('relu', name='activation_16')(batch_normalization_16)

    conv2d_14 = Conv2D(filters=48, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_14')(mixed0)
    conv2d_17 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_17')(activation_16)
    batch_normalization_14 = BatchNormalization(name='batch_normalization_14')(conv2d_14)
    batch_normalization_17 = BatchNormalization(name='batch_normalization_17')(conv2d_17)
    activation_14 = Activation('relu', name='activation_14')(batch_normalization_14)
    activation_17 = Activation('relu', name='activation_17')(batch_normalization_17)

    # average_pooling2d_2 = AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding='same',
    #                                        name='average_pooling2d_2')(mixed0)

    conv2d_13 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_13')(mixed0)
    conv2d_15 = Conv2D(filters=64, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_15')(activation_14)
    conv2d_18 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_18')(activation_17)
    conv2d_19 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_19')(mixed0)
    batch_normalization_13 = BatchNormalization(name='batch_normalization_13')(conv2d_13)
    batch_normalization_15 = BatchNormalization(name='batch_normalization_15')(conv2d_15)
    batch_normalization_18 = BatchNormalization(name='batch_normalization_18')(conv2d_18)
    batch_normalization_19 = BatchNormalization(name='batch_normalization_19')(conv2d_19)
    activation_13 = Activation('relu', name='activation_13')(batch_normalization_13)
    activation_15 = Activation('relu', name='activation_15')(batch_normalization_15)
    activation_18 = Activation('relu', name='activation_18')(batch_normalization_18)
    activation_19 = Activation('relu', name='activation_19')(batch_normalization_19)

    mixed1 = Concatenate(name='mixed1')([activation_13, activation_15, activation_18, activation_19])

    conv2d_23 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_23')(mixed1)
    batch_normalization_23 = BatchNormalization(name='batch_normalization_23')(conv2d_23)
    activation_23 = Activation('relu', name='activation_23')(batch_normalization_23)

    conv2d_21 = Conv2D(filters=48, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_21')(mixed1)
    conv2d_24 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_24')(activation_23)
    batch_normalization_21 = BatchNormalization(name='batch_normalization_21')(conv2d_21)
    batch_normalization_24 = BatchNormalization(name='batch_normalization_24')(conv2d_24)
    activation_21 = Activation('relu', name='activation_21')(batch_normalization_21)
    activation_24 = Activation('relu', name='activation_24')(batch_normalization_24)

    # average_pooling2d_3 = AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding='same',
    #                                        name='average_pooling2d_3')(mixed1)

    conv2d_20 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_20')(mixed0)
    conv2d_22 = Conv2D(filters=64, kernel_size=(1, 5), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_22')(activation_21)
    conv2d_25 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_25')(activation_24)
    conv2d_26 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_26')(mixed1)
    batch_normalization_20 = BatchNormalization(name='batch_normalization_20')(conv2d_20)
    batch_normalization_22 = BatchNormalization(name='batch_normalization_22')(conv2d_22)
    batch_normalization_25 = BatchNormalization(name='batch_normalization_25')(conv2d_25)
    batch_normalization_26 = BatchNormalization(name='batch_normalization_26')(conv2d_26)
    activation_20 = Activation('relu', name='activation_20')(batch_normalization_20)
    activation_22 = Activation('relu', name='activation_22')(batch_normalization_22)
    activation_25 = Activation('relu', name='activation_25')(batch_normalization_25)
    activation_26 = Activation('relu', name='activation_26')(batch_normalization_26)
    mixed2 = Concatenate(name='mixed2')([activation_20, activation_22, activation_25, activation_26])

    conv2d_28 = Conv2D(filters=64, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_28')(mixed2)
    batch_normalization_28 = BatchNormalization(name='batch_normalization_28')(conv2d_28)
    activation_28 = Activation('relu', name='activation_28')(batch_normalization_28)

    conv2d_29 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_29')(activation_28)
    batch_normalization_29 = BatchNormalization(name='batch_normalization_29')(conv2d_29)
    activation_29 = Activation('relu', name='activation_29')(batch_normalization_29)

    conv2d_27 = Conv2D(filters=384, kernel_size=(1, 3), activation='linear', strides=(1, 2), padding='same',
                       name='conv2d_27')(mixed2)
    conv2d_30 = Conv2D(filters=96, kernel_size=(1, 3), activation='linear', strides=(1, 2), padding='same',
                       name='conv2d_30')(activation_29)
    batch_normalization_27 = BatchNormalization(name='batch_normalization_27')(conv2d_27)
    batch_normalization_30 = BatchNormalization(name='batch_normalization_30')(conv2d_30)
    activation_27 = Activation('relu', name='activation_27')(batch_normalization_27)
    activation_30 = Activation('relu', name='activation_30')(batch_normalization_30)

    max_pooling2d_3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='same', name='max_pooling2d_3')(mixed2)

    mixed3 = Concatenate(name='mixed3')([activation_27, activation_30, max_pooling2d_3])

    conv2d_35 = Conv2D(filters=128, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_35')(mixed3)
    batch_normalization_35 = BatchNormalization(name='batch_normalization_35')(conv2d_35)
    activation_35 = Activation('relu', name='activation_35')(batch_normalization_35)

    conv2d_32 = Conv2D(filters=128, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_32')(mixed3)
    conv2d_37 = Conv2D(filters=128, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_37')(activation_35)
    batch_normalization_32 = BatchNormalization(name='batch_normalization_32')(conv2d_32)
    batch_normalization_37 = BatchNormalization(name='batch_normalization_37')(conv2d_37)
    activation_32 = Activation('relu', name='activation_32')(batch_normalization_32)
    activation_37 = Activation('relu', name='activation_37')(batch_normalization_37)

    # average_pooling2d_4 = AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding='same',
    #                                        name='average_pooling2d_3')(mixed3)

    conv2d_31 = Conv2D(filters=192, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_31')(mixed3)
    conv2d_39 = Conv2D(filters=192, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_39')(activation_32)
    conv2d_40 = Conv2D(filters=192, kernel_size=(1, 1), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_40')(activation_37)
    batch_normalization_31 = BatchNormalization(name='batch_normalization_31')(conv2d_31)
    batch_normalization_39 = BatchNormalization(name='batch_normalization_39')(conv2d_39)
    batch_normalization_40 = BatchNormalization(name='batch_normalization_40')(conv2d_40)
    activation_31 = Activation('relu', name='activation_31')(batch_normalization_31)
    activation_39 = Activation('relu', name='activation_39')(batch_normalization_39)
    activation_40 = Activation('relu', name='activation_40')(batch_normalization_40)

    mixed4 = Concatenate(name='mixed4')([activation_31, activation_39, activation_40])

    conv2d_41 = Conv2D(filters=128, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_41')(mixed4)
    batch_normalization_41 = BatchNormalization(name='batch_normalization_41')(conv2d_41)
    activation_41 = Activation('relu', name='activation_41')(batch_normalization_41)

    conv2d_42 = Conv2D(filters=96, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_42')(activation_41)
    batch_normalization_42 = BatchNormalization(name='batch_normalization_42')(conv2d_42)
    activation_42 = Activation('relu', name='activation_42')(batch_normalization_42)

    conv2d_43 = Conv2D(filters=80, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_43')(activation_42)
    batch_normalization_43 = BatchNormalization(name='batch_normalization_43')(conv2d_43)
    activation_43 = Activation('relu', name='activation_43')(batch_normalization_43)

    conv2d_44 = Conv2D(filters=64, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_44')(activation_43)
    batch_normalization_44 = BatchNormalization(name='batch_normalization_44')(conv2d_44)
    activation_44 = Activation('relu', name='activation_44')(batch_normalization_44)

    conv2d_45 = Conv2D(filters=16, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_45')(activation_44)
    batch_normalization_45 = BatchNormalization(name='batch_normalization_45')(conv2d_45)
    activation_45 = Activation('relu', name='activation_45')(batch_normalization_45)

    conv2d_46 = Conv2D(filters=4, kernel_size=(1, 7), activation='linear', strides=(1, 1), padding='same',
                       name='conv2d_46')(activation_45)
    batch_normalization_46 = BatchNormalization(name='batch_normalization_46')(conv2d_46)
    activation_46 = Activation('relu', name='activation_46')(batch_normalization_46)

    conv2d_47 = Conv2D(filters=4, kernel_size=(1, 7), activation='linear', strides=(1,1), padding='same',
                       name='conv2d_47')(activation_46)
    batch_normalization_47 = BatchNormalization(name='batch_normalization_47')(conv2d_47)
    activation_47 = Activation('relu', name='activation_47')(batch_normalization_47)

    conv2d_48 = Conv2D(filters=1, kernel_size=(1, 7), activation='linear', strides=(1, 2), padding='same',
                       name='conv2d_48')(activation_47)
    batch_normalization_48 = BatchNormalization(name='batch_normalization_48')(conv2d_48)
    activation_48= Activation('relu', name='activation_48')(batch_normalization_48)

    # conv2d_48 = Conv2D(filters=1, kernel_size=(1, 7), activation='linear', strides=(1, 2), padding='same',
    #                    name='conv2d_487')(activation_47)
    # batch_normalization_48 = BatchNormalization(name='batch_normalization_48')(conv2d_48)
    # activation_48 = Activation('relu', name='activation_48')(batch_normalization_48)

    y = Flatten()(activation_48)
    y = Dense(encoding_dim, activation='relu', )(y)
    y = Dense(encoding_dim//2, activation='relu', )(y)
    y = Dense(encoding_dim//2, activation='relu', )(y)
    y=Dropout(rate=0.2)(y)
    y = Dense(encoding_dim // 4, activation='relu', )(y)
    y = Dense(encoding_dim // 4, activation='relu', )(y)
    y = Dense(encoding_dim // 10, activation='relu', )(y)
    y = Dense(encoding_dim // 10, activation='relu', )(y)
    # y = Dense(encoding_dim // 16, activation='relu', )(y)
    # y = Dense(encoding_dim // 16, activation='relu', )(y)

    output = Dense(2, activation='sigmoid')(y)
    network = Model(input_shape, output, name="nn")
    network.summary()
    if two_task:
        first_fold = home_repo+'two-task-nn'
    else:
        first_fold = home_repo+'one-task-nn'
    aepath = first_fold+'/nn' + str(nn) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    file_raw = first_fold+'/nn' + str(nn) + '/test_conv_ae.h5'
    network.save(file_raw)
    return file_raw

make_nn(21)
#print('hey')