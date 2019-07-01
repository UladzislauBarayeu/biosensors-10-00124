import numpy as np
from keras.models import load_model
from keras import optimizers
import keras
import gc
from keras.models import model_from_json
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

        earlycallback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.0001, patience=4,
                                                      verbose=1)

        if earlystop:
            callbacks_list = [checkpoint, earlycallback]
        else:
            callbacks_list = [checkpoint]

    if optimizer is None:
        optimizer = optimizers.Adam(lr=0.00001)

    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    model_info = network.fit(train_data, labels, epochs=epoch, shuffle=True,
                             batch_size=batch_size, validation_split=validation_split, verbose=2,
                             callbacks=callbacks_list)
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

        div = [(l_max[i] - l_min[i]) for i in range(len(l_max))]
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


def save_network(name, nn ):
    aepath = '../'+home_repo + '/nn_' + str(name) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    file_raw = aepath + 'test_conv_ae.h5'
    nn.save(file_raw)


def load_train_data(subject, global_task='Task1'):
    file_path = repo_with_raw_data + global_task + '/' + str(subject) + '.json'
    json_data = open(file_path)
    d = json.load(json_data)
    json_data.close()
    x_T1 = np.expand_dims(np.array(d["T1"]["train_x"]), axis=4)
    x_T2 = np.expand_dims(np.array(d["T2"]["train_x"]), axis=4)
    train_y=np.array([keras.utils.to_categorical(d["data"]["train_y"][j]) for j in range(len(d["data"]["train_y"]))])
    return x_T1, x_T2, train_y

def load_test_data(subject,global_task='Task1'):
    file_path = repo_with_raw_data + global_task + '/' + str(subject) + '.json'
    json_data = open(file_path)
    d = json.load(json_data)
    json_data.close()

    x_T1 = np.expand_dims(np.array(d["T1"]["test_x"]), axis=4)
    x_T2 = np.expand_dims(np.array(d["T2"]["test_x"]), axis=4)

    train_y = np.array([keras.utils.to_categorical(d["data"]["test_y"][j]) for j in range(len(d["data"]["test_y"]))])
    return x_T1, x_T2, train_y

def load_minmax(subject, global_task='Task1'):
    file_path = repo_with_raw_data + global_task + '/' + str(subject) + '.json'
    json_data = open(file_path)
    d = json.load(json_data)
    json_data.close()
    minmax_T1=np.array([[d['T1']['min'][i],d['T1']['max'][i]]for i in range(len(d['T1']['max']))])
    minmax_T2=np.array([[d['T2']['min'][i],d['T2']['max'][i]]for i in range(len(d['T2']['max']))])

    return minmax_T1, minmax_T2


def load_allFalse_data(subject, global_task='Task1'):
    file_path_T1 = repo_with_raw_data + global_task + '/' + str(subject) + 'all_False_T1.json'
    json_data = open(file_path_T1)
    d_T1 = json.load(json_data)
    json_data.close()
    train_x_T1 = np.expand_dims(
        [np.reshape(d_T1["data"]["T1"]["all_false"][i]["_ArrayData_"], d_T1["data"]["T1"]["all_false"][0]["_ArraySize_"]) for i in
         range(len(d_T1["data"]["T1"]["all_false"]))], axis=4)

    file_path_T2 = repo_with_raw_data + global_task + '/' + str(subject) + 'all_False_T2.json'
    json_data = open(file_path_T2)
    d_T2 = json.load(json_data)
    json_data.close()

    train_x_T2 = np.expand_dims(
        [np.reshape(d_T2["data"]["T2"]["all_false"][i]["_ArrayData_"], d_T2["data"]["T2"]["all_false"][0]["_ArraySize_"]) for i in
         range(len(d_T2["data"]["T2"]["all_false"]))], axis=4)


    return train_x_T1, train_x_T2


#change later
def test_within_fold(s, i, global_task, file1, file2):
    test_sample_T1, test_sample_T2, test_y=load_test_data(s, global_task)
    network1 = load_network(file1)
    y_pred_1 = network1.predict(test_sample_T1)

    network2 = load_network(file2)
    y_pred_2 = network2.predict(test_sample_T2)

    true_values_right_T1 = 0
    false_values_right_T1 = 0

    true_values_right_T2 = 0
    false_values_right_T2 = 0

    true_values_right_both = 0
    false_values_right_both = 0

    len_true_all = 0
    len_false_all = 0
    len_true_all += ((test_y == true_vector).sum() // 2)
    len_false_all += ((test_y == false_vector).sum() // 2)

    for j in range(len(test_y)):

        if (y_pred_1[j][0] > 0.5):
            t1 = [1.0, 0.0]
        else:
            t1 = [0.0, 1.0]

        if (y_pred_2[j][0] > 0.5):
            t2 = [1.0, 0.0]
        else:
            t2 = [0.0, 1.0]

        if (test_y[j] == true_vector).all():
            # check T1 task
            if (t1 == test_y[j]).all():
                true_values_right_T1 += 1
            # check T2 task
            if (t2 == test_y[j]).all():
                true_values_right_T2 += 1
            # check both tasks
            if (t1 == test_y[j]).all() and (t2 == test_y[j]).all():
                true_values_right_both += 1

        else:
            # check T1 task
            if (t1 == test_y[j]).all():
                false_values_right_T1 += 1
            # check T2 task
            if (t2 == test_y[j]).all():
                false_values_right_T2 += 1
            # check both tasks
            if ((t1 == test_y[j]).all() and (t2 == test_y[j]).all()) or (
                    (t1 == test_y[j]).all() and (t2 == true_vector).all()) or (
                    (t1 == true_vector).all() and (t2 == test_y[j]).all()):
                false_values_right_both += 1

    print('For T1:  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
        (1 - true_values_right_T1 / len_true_all), (1 - false_values_right_T1 / len_false_all)))

    print('For T2  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
        (1 - true_values_right_T2 / len_true_all), (1 - false_values_right_T2 / len_false_all)))

    print('For both  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
        (1 - true_values_right_both / len_true_all), (1 - false_values_right_both / len_false_all)))

    print(
        'For T1 accuracy overall: {} '.format((true_values_right_T1 + false_values_right_T1) / (len_true_all + len_false_all)))

    print(
        'For T2  accuracy overall {} '.format((true_values_right_T2 + false_values_right_T2) / (len_true_all + len_false_all)))

    print('For both  accuracy overall {} '.format((true_values_right_both + false_values_right_both) / (len_true_all + len_false_all)))


def test_within_fold_allFalse(s,i, global_task, file1, file2):

    all_T1, all_T2, test_y=load_allFalse_data(s, global_task)

    network1 = load_network(file1)
    y_pred_1 = network1.predict(all_T1)

    network2 = load_network(file2)
    y_pred_2 = network2.predict(all_T2)

    true_values_right_T1 = 0
    false_values_right_T1 = 0

    true_values_right_T2 = 0
    false_values_right_T2 = 0

    true_values_right_both = 0
    false_values_right_both = 0

    len_true_all = 0
    len_false_all = 0
    len_true_all += ((test_y == true_vector).sum() // 2)
    len_false_all += ((test_y == false_vector).sum() // 2)

    for j in range(len(test_y)):

        if (y_pred_1[j][0] > 0.5):
            t1 = [1.0, 0.0]
        else:
            t1 = [0.0, 1.0]

        if (y_pred_2[j][0] > 0.5):
            t2 = [1.0, 0.0]
        else:
            t2 = [0.0, 1.0]

        if (test_y[j] == true_vector).all():
            # check T1 task
            if (t1 == test_y[j]).all():
                true_values_right_T1 += 1
            # check T2 task
            if (t2 == test_y[j]).all():
                true_values_right_T2 += 1
            # check both tasks
            if (t1 == test_y[j]).all() and (t2 == test_y[j]).all():
                true_values_right_both += 1

        else:
            # check T1 task
            if (t1 == test_y[j]).all():
                false_values_right_T1 += 1
            # check T2 task
            if (t2 == test_y[j]).all():
                false_values_right_T2 += 1

            # check both tasks
            if ((t1 == test_y[j]).all() and (t2 == test_y[j]).all()) or (
                    (t1 == test_y[j]).all() and (t2 == true_vector).all()) or (
                    (t1 == true_vector).all() and (t2 == test_y[j]).all()):
                false_values_right_both += 1

    print('For T1:  \n type II error in cross-validation: {} \n '.format(
        1 - false_values_right_T1 / len_false_all))

    print('For T2  \n  type II error in cross-validation: {} \n '.format(
        (1 - false_values_right_T2 / len_false_all)))

    print('For both  \n type II error in cross-validation: {} \n '.format(
        (1 - false_values_right_both / len_false_all)))


if __name__ == '__main__':
    load_minmax(1, global_task)