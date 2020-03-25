from keras.models import load_model
from keras import *
import numpy as np
import json
from EEG_class import *
from network_utils import *
from configurations import *
import h5py


def export_nn_for_svm_two_tasks(nn, s, global_task='Task1', channels='16_channels'):
    aepath = home_repo + '/'+channels+'/nn_' + str(nn) + '/' + global_task + '/' + str(s) + '/'

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        test_x_1 = f["test_sample_T1"][:]
        test_x_2 = f["test_sample_T2"][:]
        test_labels = f["test_labels"][:]

        train_x_1 = f["train_sample_T1"][:]
        train_x_2 = f["train_sample_T2"][:]
        train_labels = f["train_labels"][:]

    number_of_folds = train_labels.shape[0]

    t1_train_data_predicted = [0 for i in range(number_of_folds)]
    t1_test_data_predicted = [0 for i in range(number_of_folds)]

    t2_train_data_predicted = [0 for i in range(number_of_folds)]
    t2_test_data_predicted = [0 for i in range(number_of_folds)]

    test_y = [0 for i in range(number_of_folds)]
    train_y = [0 for i in range(number_of_folds)]

    for fold in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'
        layer_name = 'flatten_1'
        model1 = load_network(file1)
        intermediate_layer_model1 = Model(inputs=model1.input,
                                          outputs=model1.get_layer(layer_name).output)
        model2 = load_network(file2)
        intermediate_layer_model2 = Model(inputs=model2.input,
                                          outputs=model2.get_layer(layer_name).output)

        test_y[fold] = test_labels[fold].tolist()
        train_y[fold] = train_labels[fold].tolist()

        train_data_1 = intermediate_layer_model1.predict(train_x_1[fold])
        train_data_2 = intermediate_layer_model2.predict(train_x_2[fold])
        test_data_1 = intermediate_layer_model1.predict(test_x_1[fold])
        test_data_2 = intermediate_layer_model2.predict(test_x_2[fold])

        t1_train_data_predicted[fold] = train_data_1.tolist()
        t2_train_data_predicted[fold] = train_data_2.tolist()

        t1_test_data_predicted[fold] = test_data_1.tolist()
        t2_test_data_predicted[fold] = test_data_2.tolist()

    jsondic = {'T1': {'train_sample': t1_train_data_predicted, 'test_sample': t1_test_data_predicted},
               'T2': {'train_sample': t2_train_data_predicted, 'test_sample': t2_test_data_predicted},
               'train_y': train_y, 'test_y': test_y}

    dir_for_output = matlab_repo_for_saving_svm+channels+'/nn_' + str(nn) + '/' + str(global_task)

    if not os.path.exists(dir_for_output):
        os.makedirs(dir_for_output)
    outfile = open(dir_for_output + '/data_for_svm_s' + str(s) + '.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()


def create_json_for_ROC(nn, s, channels='16_channels'):
    
    file_repo = python_repo_for_saving_predicted +'/'+channels+'/nn_' + str(nn) + '/' + str(global_task) + '/predicted_data_s' + str(
        s) + '.h5'

    with h5py.File(file_repo, 'r') as f:
        t1_test_data = f["T1_predicted"][:, :, 0]
        t2_test_data = f["T2_predicted"][:, :, 0]
        test_y_all = f["test_labels"][:, :, 0]

    labels = [['' for i in range(test_y_all.shape[1])] for j in range(test_y_all.shape[0])]

    for fold in range(test_y_all.shape[0]):
        for trial in range(test_y_all.shape[1]):
            if test_y_all[fold][trial] == 1:
                labels[fold][trial] = 'subject'
            else:
                labels[fold][trial] = 'other'

    jsondic = {'T1': t1_test_data.tolist(),
               'T2': t2_test_data.tolist(),
               'labels': labels}

    dir_for_output = matlab_repo_for_saving_svm +channels+'/nn_' + str(nn) + '/' + str(global_task)

    if not os.path.exists(dir_for_output):
        os.makedirs(dir_for_output)

    outfile = open(dir_for_output + '/predicted_data_for_ROC_s' + str(s) + '.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()


def export_allFalse_for_svm_two_tasks(nn, s, global_task='Task1', channels='16_channels'):
    aepath = home_repo +'/'+channels+'/nn_'  + str(nn) + '/' + global_task + '/' + str(s) + '/'

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        minmax_T1 = f["minmax_T1"][:]
        minmax_T2 = f["minmax_T2"][:]

    number_of_folds = minmax_T1.shape[0]

    t1_test_data_predicted = [0 for i in range(number_of_folds)]
    t2_test_data_predicted = [0 for i in range(number_of_folds)]
    test_y = [0 for i in range(number_of_folds)]

    all_T1, all_T2 = load_allFalse(s, global_task, channels=channels)

    for fold in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'
        layer_name = 'flatten_1'
        model1 = load_network(file1)
        normalized_T1 = normalize_data(all_T1, minmax_T1[fold])
        normalized_T2 = normalize_data(all_T2, minmax_T2[fold])

        intermediate_layer_model1 = Model(inputs=model1.input,
                                          outputs=model1.get_layer(layer_name).output)
        model2 = load_network(file2)
        intermediate_layer_model2 = Model(inputs=model2.input,
                                          outputs=model2.get_layer(layer_name).output)

        test_y[fold] = [[0.0, 1.0] for l in range(len(normalized_T1))]
        test_data_1 = intermediate_layer_model1.predict(normalized_T1)
        test_data_2 = intermediate_layer_model2.predict(normalized_T2)

        t1_test_data_predicted[fold] = test_data_1.tolist()
        t2_test_data_predicted[fold] = test_data_2.tolist()

    jsondic = {'T1': {'test_sample': t1_test_data_predicted},
               'T2': {'test_sample': t2_test_data_predicted}, 'test_y': test_y}

    dir_for_output = matlab_repo_for_saving_svm +channels+'/nn_'  +str(nn) + '/' + str(global_task) + '/'

    if not os.path.exists(dir_for_output):
        os.makedirs(dir_for_output)
    outfile = open(dir_for_output + 'predicted_data_for_SVM_all_false_subjects_s' + str(s) + '.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()

