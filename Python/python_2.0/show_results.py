import os
import random
import matplotlib.pyplot as plt
import h5py
import numpy as np
true_vector=np.array([1.0, 0.0])
false_vector=np.array([0.0, 1.0])
import json
from keras import *
from network_utils import normalize_data, load_network


def predict_two_tasks(nn, s, number_of_folds=5):

    aepath = 'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'


    t1_test_data=[0 for i in range(number_of_folds)]
    t2_test_data=[0 for i in range(number_of_folds)]
    test_y= [0 for i in range(number_of_folds)]

    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        test_x_1 = f["test_sample_T1"][:]
        test_x_2= f["test_sample_T2"][:]
        test_labels= f["test_labels"][:]

    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i) + '.h5'

        model1 = load_network(file1)
        model2 = load_network(file2)

        t1_test_data[i] = model1.predict(test_x_1[i])
        t2_test_data[i] = model2.predict(test_x_2[i])

    with h5py.File(aepath + 'predicted_data.h5', 'w') as f:
        d = f.create_dataset("T1_predicted", data=np.array(t1_test_data, dtype=np.float64))
        d = f.create_dataset("T2_predicted", data=np.array(t2_test_data, dtype=np.float64))
        d = f.create_dataset("test_labels", data=np.array(test_labels, dtype=np.float64))


def read_predicted_file(nn, s, file='predicted_data.h5', all_false=False):
    aepath = 'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'

    sum_false_right_both = 0
    sum_true_right_both = 0
    sum_false_right_T1 = 0
    sum_true_right_T1 = 0
    sum_false_right_T2 = 0
    sum_true_right_T2 = 0
    len_of_false = 0
    len_of_true = 0
    with h5py.File(aepath + file, 'r') as f:
        t1_test_data = f["T1_predicted"][:]
        t2_test_data= f["T2_predicted"][:]
        test_y_all= f["test_labels"][:]


    for i in range(t1_test_data.shape[0]):

        y_pred_1=t1_test_data[i]
        y_pred_2 = t2_test_data[i]
        test_y=test_y_all[i]
        len_true = ((test_y == true_vector).sum() // 2)
        len_false = ((test_y == false_vector).sum() // 2)

        true_values_right_both = 0
        false_values_right_both = 0

        true_values_right_T1 = 0
        false_values_right_T1 = 0

        true_values_right_T2 = 0
        false_values_right_T2 = 0

        for j in range(len(test_y)):

            if (y_pred_1[j][0] >=0.8):
                t1 = [1.0, 0.0]
            else:
                t1 = [0.0, 1.0]

            if (y_pred_2[j][0] >= 0.8):
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

        if len_true > 0:
            sum_true_right_T1 +=true_values_right_T1
            sum_true_right_T2+=true_values_right_T2
            sum_true_right_both+=true_values_right_both
            len_of_true+=len_true

        if len_false > 0:
            sum_false_right_T1 += false_values_right_T1
            sum_false_right_T2 += false_values_right_T2
            sum_false_right_both += false_values_right_both
            len_of_false += len_false

    return sum_true_right_T1, sum_false_right_T1, sum_true_right_T2, sum_false_right_T2, sum_true_right_both, sum_false_right_both, len_of_true, len_of_false


def mean_accuracy(nn, subjects, number_of_folds=5):

    sum_true_right_T1 = [0 for i in range(len(subjects))]
    sum_false_right_T1 = [0 for i in range(len(subjects))]
    sum_true_right_T2 = [0 for i in range(len(subjects))]
    sum_false_right_T2 = [0 for i in range(len(subjects))]
    sum_true_right_both = [0 for i in range(len(subjects))]
    sum_false_right_both = [0 for i in range(len(subjects))]
    len_of_true = [0 for i in range(len(subjects))]
    len_of_false = [0 for i in range(len(subjects))]

    for i in range(len(subjects)):
        sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i] = read_predicted_file(nn, subjects[i])

    true_right_T1 = sum(sum_true_right_T1)
    false_right_T1 = sum(sum_false_right_T1)

    true_right_T2 = sum(sum_true_right_T2)
    false_right_T2 = sum(sum_false_right_T2)

    true_right_both = sum(sum_true_right_both)
    false_right_both = sum(sum_false_right_both)
    len_of_true_sum = sum(len_of_true)
    len_of_false_sum = sum(len_of_false)
    print("===================================================\n OVERALL")
    print('For T1:  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        (1 - true_right_T1 / len_of_true_sum), (1 - false_right_T1 / len_of_false_sum)))

    print('For T2  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        (1 - true_right_T2 / len_of_true_sum), (1 - false_right_T2 / len_of_false_sum)))

    print('For both  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        (1 - true_right_both / len_of_true_sum), (1 - false_right_both / len_of_false_sum)))

    print(
        'For T1 accuracy overall: {} '.format((true_right_T1 + false_right_T1) / (len_of_true_sum + len_of_false_sum)))

    print(
        'For T2  accuracy overall {} '.format((true_right_T2 + false_right_T2) / (len_of_true_sum + len_of_false_sum)))

    print('For both  accuracy overall {} '.format(
        (true_right_both + false_right_both) / (len_of_true_sum + len_of_false_sum)))


#predict_two_tasks(21, 65)
#predict_two_tasks(21, 75)
#predict_two_tasks(21, 85)
#predict_two_tasks(21, 55)
mean_accuracy(21, [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])