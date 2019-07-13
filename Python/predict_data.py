import json
import os
import numpy as np
import h5py
from EEG_class import *
from keras import *
from network_utils import normalize_data, load_network, load_allFalse
from configurations import *

def predict_two_tasks(nn, s, global_task='Task1'):

    aepath = home_repo+'nn_' + str(nn) + '/' +global_task+'/'+ str(s) + '/'

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        test_x_1 = f["test_sample_T1"][:]
        test_x_2= f["test_sample_T2"][:]
        test_labels= f["test_labels"][:]


    number_of_folds = test_labels.shape[0]
    t1_test_data=[0 for i in range(number_of_folds)]
    t2_test_data=[0 for i in range(number_of_folds)]

    for fold in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'

        model1 = load_network(file1)
        model2 = load_network(file2)

        t1_test_data[fold] = model1.predict(test_x_1[fold])
        t2_test_data[fold] = model2.predict(test_x_2[fold])

    with h5py.File(aepath + 'predicted_data.h5', 'w') as f:
        d = f.create_dataset("T1_predicted", data=np.array(t1_test_data, dtype=np.float64))
        d = f.create_dataset("T2_predicted", data=np.array(t2_test_data, dtype=np.float64))
        d = f.create_dataset("test_labels", data=np.array(test_labels, dtype=np.float64))


def predict_allFalse_two_tasks(nn, s, global_task='Task1'):

    aepath = home_repo+'nn_' + str(nn) + '/' +global_task+'/'+ str(s) + '/'

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        minmax_T1 = f["minmax_T1"][:]
        minmax_T2 = f["minmax_T2"][:]

    number_of_folds = minmax_T1.shape[0]
    t1_test_data_predicted = [0 for i in range(number_of_folds)]
    t2_test_data_predicted = [0 for i in range(number_of_folds)]
    test_y = [0 for i in range(number_of_folds)]

    all_T1 , all_T2 = load_allFalse(s, global_task)

    for fold in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'
        model1 = load_network(file1)
        model2 = load_network(file2)

        normalized_T1 = normalize_data(all_T1, minmax_T1[fold])
        normalized_T2 = normalize_data(all_T2, minmax_T2[fold])

        test_y[fold]=[[0.0, 1.0] for i in range(len(normalized_T1))]
        test_data_1 = model1.predict(normalized_T1)
        test_data_2 = model2.predict(normalized_T2)

        t1_test_data_predicted[fold] = test_data_1.tolist()
        t2_test_data_predicted[fold] = test_data_2.tolist()


    with h5py.File(aepath + 'predicted_data_all_false.h5', 'w') as f:
        d = f.create_dataset("T1_predicted", data=np.array(t1_test_data_predicted, dtype=np.float64))
        d = f.create_dataset("T2_predicted", data=np.array(t2_test_data_predicted, dtype=np.float64))
        d = f.create_dataset("test_labels", data=np.array(test_y, dtype=np.float64))


if __name__ == '__main__':
    predict_allFalse_two_tasks("simple_1_with_dropout_2", 15)


