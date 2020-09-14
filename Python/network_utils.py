# ================================================
# Author: Nastassya Horlava
# Github: @HorlavaNastassya
# Email: g.nasta.work@gmail.com
# ===============================================

from keras.models import load_model
from keras import optimizers
import keras
import json
from configurations import *
import os


def load_network(file, name=None):
    model_cpu = load_model(file)

    if name is not None:
        model_cpu.name = name
    return model_cpu


def train_nn(h5file, train_data, labels, batch_size=32, callbacks_list=None, optimizer=None,
             epoch=2, ae_name='test_ae.h5',
             loss='mean_squared_error', period=2, validation_split=0.2, earlystop=False):
    network = load_network(h5file)

    if callbacks_list is None:

        checkpoint = keras.callbacks.ModelCheckpoint(ae_name, monitor='val_loss', verbose=0, save_best_only=True,
                                                     save_weights_only=False, mode='auto', period=period)

        earlycallback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5000,
                                                      verbose=1)

        if earlystop:
            callbacks_list = [checkpoint, earlycallback]
        else:
            callbacks_list = [checkpoint]

    if optimizer is None:
        optimizer = optimizers.Adam(lr=0.0001)

    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_info = network.fit(train_data, labels, epochs=epoch, shuffle=True,
                             batch_size=batch_size, validation_split=validation_split, verbose=2,
                             callbacks=callbacks_list)
    return model_info


def normalize_data(x_array, minmax_arr=None):
    out = np.zeros(x_array.shape, x_array.dtype)

    if minmax_arr is None:
        tmp = lambda arr, feature_row: arr[:, feature_row, :, :]
        l_min = []
        l_max = []
        div = []
        for feature_row in range(x_array.shape[1]):
            loc_min = np.min(tmp(x_array, feature_row))
            loc_max = np.max(tmp(x_array, feature_row))
            l_min.append(loc_min)
            l_max.append(loc_max)
            div_real = loc_max - loc_min
            div.append(div_real)

        tmp = lambda arr, trial, feature_row: arr[trial, feature_row, :, :]

        for trial in range(x_array.shape[0]):
            for feature_row in range(x_array.shape[1]):
                o = tmp(out, trial, feature_row)
                o[:] = (tmp(x_array, trial, feature_row) - l_min[feature_row]) / div[feature_row]

        return out, np.array([l_min, l_max])

    else:

        l_min = minmax_arr[0]
        l_max = minmax_arr[1]

        div = [(l_max[i] - l_min[i]) for i in range(len(l_max))]

        tmp = lambda mas, i, j: mas[i, j, :, :]

        for trial in range(x_array.shape[0]):
            for feature_row in range(x_array.shape[1]):
                o = tmp(out, trial, feature_row)
                o[:] = (tmp(x_array, trial, feature_row) - l_min[feature_row]) / div[feature_row]

        for trial in range(out.shape[0]):
            for feature_row in range(out.shape[1]):
                for el in range(out.shape[2]):
                    for t in range(out.shape[3]):
                        if (out[trial][feature_row][el][t] > 1.0):
                            out[trial][feature_row][el][t] = 1.0

                        if (out[trial][feature_row][el][t] < 0.0):
                            out[trial][feature_row][el][t] = 0.0
        return out

def normalize_data_one_channel(x_array, minmax_arr=None):
    out = np.zeros(x_array.shape, x_array.dtype)

    if minmax_arr is None:

        l_min = []
        l_max = []
        div = []
        for i, trial in enumerate(x_array):
            loc_min = np.min(trial)
            loc_max = np.max(trial)
            l_min.append(loc_min)
            l_max.append(loc_max)
            div_real = loc_max - loc_min
            div.append(div_real)
            out[i]=(trial-loc_min)/div_real
        return out, np.array([l_min, l_max])

    else:

        l_min = minmax_arr[0]
        l_max = minmax_arr[1]

        div = [(l_max[i] - l_min[i]) for i in range(len(l_max))]



        for i, trial in enumerate(x_array):
            out[i] = (trial - l_min[i]) / div[i]


        for trial in range(out.shape[0]):
            for feature_row in range(out.shape[1]):
                if (out[trial][feature_row] > 1.0):
                    out[trial][feature_row] = 1.0

                if (out[trial][feature_row] < 0.0):
                    out[trial][feature_row] = 0.0
        return out

def denorm_data(x_array, minmax_arr):
    out = np.zeros(x_array.shape, x_array.dtype)
    l_min = minmax_arr[0]
    l_max = minmax_arr[1]
    div = [l_max[i] - l_min[i] for i in range(len(l_max))]
    tmp = lambda mas, trial, feature_row: mas[trial, feature_row, :, :]
    for i in range(x_array.shape[0]):
        for j in range(x_array.shape[1]):
            o = tmp(out, i, j)
            o[:] = (tmp(x_array, i, j) * div[j] + l_min[j])
    return out


def show_model(file, out_file):
    model = load_network(file)
    from keras.utils import plot_model
    plot_model(model, to_file=out_file, show_layer_names=False, rankdir='LR')


def save_network(name, nn, additional_folder_for_nn='', channels='16_channels'):
    aepath = additional_folder_for_nn + home_repo + str(channels)+'/nn_' + str(name) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    file_raw = aepath + 'test_conv_ae.h5'
    nn.save(file_raw)


def load_allFalse(subject, global_task='Task1', channels='16_channels'):
    all_T1 = []
    all_T2 = []

    for falseSubject in range(1, number_of_subjects+1, 1):
        if falseSubject != subject:
            directory = repo_with_raw_data + global_task + "/"+channels+'/'
            h5file = str(falseSubject) + '.json'
            path = directory + h5file
            json_data = open(path)
            d = json.load(json_data)
            json_data.close()
            data_T1 = np.array(d['T1'][:number_of_trials])
            data_T1 = np.transpose(data_T1, (0, 2, 1))
            data_T1 = np.expand_dims(data_T1, axis=3)
            all_T1.extend(data_T1)
            data_T2 = np.array(d['T2'][:number_of_trials])
            data_T2 = np.transpose(data_T2, (0, 2, 1))
            data_T2 = np.expand_dims(data_T2, axis=3)
            all_T2.extend(data_T2)

    all_T1 = np.array(all_T1)
    all_T2 = np.array(all_T2)
    return all_T1, all_T2


