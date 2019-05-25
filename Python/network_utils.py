import numpy as np
from keras.models import load_model
from keras import optimizers
import keras
import gc
from keras.models import model_from_json
import json

def load_network(file, name=None, format="h5"):
    '''
    load network from file, give it name(optional)
    return network and attribute 'use_channels',
    if use_channels=False, then real and imaginary parts of image are treated as separate images,
    if use_channels=True, then real and imaginary parts of image are treated as channels in one image
    '''
    if format == "h5":
        model_cpu = load_model(file)
    if format == "json":
        json_data = open(file)
        nn = json.load(json_data)
        json_data.close()
        model_cpu = model_from_json(nn)

    if name is not None:
        model_cpu.name = name
    return model_cpu


def train_autoencoder(h5file, train_data, labels, format="h5", batch_size=32, callbacks_list=None, optimizer=None,
                      epoch=2, ae_name='test_ae.h5',
                      loss='mean_squared_error', period=2, validation_split=0.2, earlystop=False):
    network = load_network(h5file, format=format)

    if callbacks_list is None:

        checkpoint = keras.callbacks.ModelCheckpoint(ae_name, monitor='val_loss', verbose=0, save_best_only=True,
                                                     save_weights_only=False, mode='auto', period=period)

        earlycallback = keras.callbacks.EarlyStopping(monitor='loss', mode='auto', min_delta=0.00000001, patience=1,
                                                      verbose=1, )
        # callbacks_list = [checkpoint, earlycallback]

        callbacks_list = [checkpoint]
        if earlystop:
            callbacks_list = [earlycallback]
        # callbacks_list = [checkpoint]
        #

    if optimizer is None:
        optimizer = optimizers.Adam(lr=0.00001)

    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_info = network.fit(train_data, labels, epochs=epoch, shuffle=True,
                             batch_size=batch_size, validation_split=validation_split, verbose=2,
                             callbacks=callbacks_list)
    # if save_ae:
    #     network.save(ae_name)
    return model_info


def normalize_data(x_array, minmax_arr=None):
    out = np.zeros(x_array.shape, x_array.dtype)

    if minmax_arr is None:
        tmp = lambda a, i: a[:, i, :, :]
        size = x_array.shape[1]
        l_min = []
        l_max = []
        div = []
        for i in range(size):
            loc_min = np.min(tmp(x_array, i))
            loc_max = np.max(tmp(x_array, i))
            l_min.append(loc_min)
            l_max.append(loc_max)
            div_real = loc_max - loc_min
            div.append(div_real)

        tmp = lambda a, i, j: a[i, j, :, :]
        for i in range(x_array.shape[0]):
            for j in range(x_array.shape[1]):
                o = tmp(out, i, j)
                o[:] = (tmp(x_array, i, j) - l_min[j]) / div[j]

        return out, np.array([l_min, l_max])
    else:

        l_min = minmax_arr[0]
        l_max = minmax_arr[1]

        div = [l_max[i] - l_min[i] for i in range(len(l_max))]
        tmp = lambda a, i, j: a[i, j, :, :]
        for i in range(x_array.shape[0]):
            for j in range(x_array.shape[1]):
                o = tmp(out, i, j)
                o[:] = (tmp(x_array, i, j) - l_min[j]) / div[j]

        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                for k in range(out.shape[2]):
                    for t in range(out.shape[3]):
                        if (out[i][j][k][t] > 1.0):
                            out[i][j][k][t] = 1.0

                        if (out[i][j][k][t] < 0.0):
                            out[i][j][k][t] = 0.0
        return out


def denorm_data(x_array, minmax_arr):
    out = np.zeros(x_array.shape, x_array.dtype)
    l_min = minmax_arr[0]
    l_max = minmax_arr[1]
    div = [l_max[i] - l_min[i] for i in range(len(l_max))]
    tmp = lambda a, i, j: a[i, j, :, :]
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            o = tmp(out, i, j)
            o[:] = (tmp(x_array, i, j) * div[j] + l_min[j])
    return out


def show_model(file, out_file):
    model = load_network(file)
    from keras.utils import plot_model
    plot_model(model, to_file=out_file, show_layer_names=False, rankdir='LR')
