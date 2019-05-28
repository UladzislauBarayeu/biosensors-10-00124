from keras.models import load_model
from keras import *
import numpy as np
import json
from EEG_class import *
from network_utils import *
from configurations import *

def export_nn_for_svm_two_tasks(nn, s, global_task='Task1'):
    aepath = home_repo+'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        test_x_1 = f["test_sample_T1"][:]
        test_x_2 = f["test_sample_T2"][:]
        test_labels = f["test_labels"][:]

        train_x_1 = f["train_sample_T1"][:]
        train_x_2 = f["train_sample_T2"][:]
        train_labels = f["train_labels"][:]

    number_of_folds = train_labels.shape[0]

    test1 = EEGdata()
    file = str(s) + '.json'
    all_labels=test1.load_labels(file, global_task=global_task)
    labels=['' for i in range(48)]
    j=0

    for i in range(0, 1536, 64):
        labels[j] = all_labels[i][5:]
        j += 1
        labels[j] = all_labels[i][5:]
        j += 1

    t1_train_data_predicted=[0 for i in range(number_of_folds)]
    t1_test_data_predicted=[0 for i in range(number_of_folds)]

    t2_train_data_predicted=[0 for i in range(number_of_folds)]
    t2_test_data_predicted=[0 for i in range(number_of_folds)]

    test_y= [0 for i in range(number_of_folds)]
    train_y= [0 for i in range(number_of_folds)]


    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i) + '.h5'
        layer_name = 'flatten_1'
        model1 = load_network(file1)
        intermediate_layer_model1 = Model(inputs=model1.input,
                                         outputs=model1.get_layer(layer_name).output)
        model2 = load_network(file2)
        intermediate_layer_model2 = Model(inputs=model2.input,
                                          outputs=model2.get_layer(layer_name).output)

        test_y[i]=test_labels[i].tolist()
        train_y[i]=train_labels[i].tolist()

        train_data_1 = intermediate_layer_model1.predict(train_x_1[i])
        train_data_2 = intermediate_layer_model2.predict(train_x_2[i])
        test_data_1 = intermediate_layer_model1.predict(test_x_1[i])
        test_data_2 = intermediate_layer_model2.predict(test_x_2[i])

        t1_train_data_predicted[i] = train_data_1.tolist()
        t2_train_data_predicted[i] = train_data_2.tolist()

        t1_test_data_predicted[i] = test_data_1.tolist()
        t2_test_data_predicted[i] = test_data_2.tolist()



    jsondic = {'T1':{'train_sample':t1_train_data_predicted,'test_sample':t1_test_data_predicted},
               'T2': {'train_sample': t2_train_data_predicted, 'test_sample': t2_test_data_predicted},
               'result_label':labels, 'train_y': train_y, 'test_y': test_y}

    dir_for_output=matlab_repo_for_saving_svm+str(nn)+'/'
    if not os.path.exists(dir_for_output):
        os.makedirs(dir_for_output)
    outfile = open(dir_for_output+'data_for_svm_s'+str(s)+'.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()

def create_json_for_ROC(nn, s, file='predicted_data.h5'):

    aepath = home_repo+'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'
    with h5py.File(aepath + file, 'r') as f:
        t1_test_data = f["T1_predicted"][:,:, 0]
        t2_test_data = f["T2_predicted"][:, :,0]
        test_y_all = f["test_labels"][:,:, 0]

    labels=[['' for i in range (test_y_all.shape[1])] for j in range(test_y_all.shape[0])]

    for i in range(test_y_all.shape[0]):
        for j in range(test_y_all.shape[1]):
            if test_y_all[i][j]==1:
                labels[i][j]='subject'
            else:
                labels[i][j] = 'other'

    jsondic = {'T1': t1_test_data.tolist(),
               'T2': t2_test_data.tolist(),
               'labels': labels}
    outfile = open(home_repo+'two-task-nn/nn' + str(nn) + '/predicted_data_for_ROC_s' + str(s) + '.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()


def export_allFalse_for_svm_two_tasks(nn, s, global_task='Task1'):
    aepath = home_repo+'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'

    test1 = EEGdata()
    file = str(s) + '.json'
    all_labels=test1.load_labels(file, global_task=global_task)
    labels=['' for i in range(48)]
    j=0

    for i in range(0, 1536, 64):
        labels[j] = all_labels[i][5:]
        j += 1
        labels[j] = all_labels[i][5:]
        j += 1

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        minmax_T1 = f["minmax_T1"][:]
        minmax_T2 = f["minmax_T2"][:]
    number_of_folds = minmax_T1.shape[0]

    t1_test_data_predicted=[0 for i in range(number_of_folds)]
    t2_test_data_predicted=[0 for i in range(number_of_folds)]
    test_y = [0 for i in range(number_of_folds)]

    all_T1 = []
    all_T2 = []
    for j in range(1, 106, 1):
        if j != s:
            directory = repo_with_raw_data+global_task+"/"
            h5file = str(s) + '.json'
            path = directory + h5file
            json_data = open(path)
            d = json.load(json_data)
            json_data.close()
            data_T1 = np.array(d['T1'][:110])
            data_T1 = np.transpose(data_T1, (0, 2, 1))
            data_T1 = np.expand_dims(data_T1, axis=3)
            all_T1.extend(data_T1)
            data_T2 = np.array(d['T2'][:110])
            data_T2 = np.transpose(data_T2, (0, 2, 1))
            data_T2 = np.expand_dims(data_T2, axis=3)
            all_T2.extend(data_T2)

    all_T1 = np.array(all_T1)
    all_T2 = np.array(all_T2)


    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i) + '.h5'
        layer_name = 'flatten_1'
        model1 = load_network(file1)
        normalized_T1 = normalize_data(all_T1, minmax_T1[i])
        normalized_T2 = normalize_data(all_T2, minmax_T2[i])

        intermediate_layer_model1 = Model(inputs=model1.input,
                                         outputs=model1.get_layer(layer_name).output)
        model2 = load_network(file2)
        intermediate_layer_model2 = Model(inputs=model2.input,
                                          outputs=model2.get_layer(layer_name).output)

        test_y[i]=[[0.0, 1.0] for l in range(len(normalized_T1))]
        test_data_1 = intermediate_layer_model1.predict(normalized_T1)
        test_data_2 = intermediate_layer_model2.predict(normalized_T2)

        t1_test_data_predicted[i] = test_data_1.tolist()
        t2_test_data_predicted[i] = test_data_2.tolist()

    jsondic = {'T1':{'test_sample':t1_test_data_predicted},
               'T2': {'test_sample': t2_test_data_predicted},
               'result_label':labels, 'test_y': test_y}


    dir_for_output = matlab_repo_for_saving_all_false_svm + str(nn) + '/'

    if not os.path.exists(dir_for_output):
        os.makedirs(dir_for_output)
    outfile = open(dir_for_output + 'predicted_data_for_SVM_all_false_subjects_s' + str(s) + '.json', 'w')
    json.dump(jsondic, outfile)
    outfile.close()


if __name__=="__main__":
    export_nn_for_svm_two_tasks(211, 5)