import h5py
from configurations import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def read_predicted_file(nn, s, file='predicted_data.h5', threshold_for_T1=0.5, threshold_for_T2=0.5):

    aepath = home_repo + 'nn_' + str(nn) + '/' + global_task + '/' + str(s) + '/'
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
        t2_test_data = f["T2_predicted"][:]
        test_y_all = f["test_labels"][:]

    for fold in range(t1_test_data.shape[0]):

        y_pred_1 = t1_test_data[fold]
        y_pred_2 = t2_test_data[fold]
        test_y = test_y_all[fold]
        len_true = ((test_y == true_vector).sum() // 2)
        len_false = ((test_y == false_vector).sum() // 2)

        true_values_right_both = 0
        false_values_right_both = 0

        true_values_right_T1 = 0
        false_values_right_T1 = 0

        true_values_right_T2 = 0
        false_values_right_T2 = 0

        for trial in range(len(test_y)):

            if (y_pred_1[trial][0] > threshold_for_T1):
                t1 = [1.0, 0.0]
            else:
                t1 = [0.0, 1.0]

            if (y_pred_2[trial][0] > threshold_for_T2):
                t2 = [1.0, 0.0]
            else:
                t2 = [0.0, 1.0]

            if (test_y[trial] == true_vector).all():
                # check T1 task
                if (t1 == test_y[trial]).all():
                    true_values_right_T1 += 1
                # check T2 task
                if (t2 == test_y[trial]).all():
                    true_values_right_T2 += 1
                # check both tasks
                if (t1 == test_y[trial]).all() and (t2 == test_y[trial]).all():
                    true_values_right_both += 1

            else:
                # check T1 task
                if (t1 == test_y[trial]).all():
                    false_values_right_T1 += 1
                # check T2 task
                if (t2 == test_y[trial]).all():
                    false_values_right_T2 += 1
                # check both tasks
                if ((t1 == test_y[trial]).all() and (t2 == test_y[trial]).all()) or (
                        (t1 == test_y[trial]).all() and (t2 == true_vector).all()) or (
                        (t1 == true_vector).all() and (t2 == test_y[trial]).all()):
                    false_values_right_both += 1

        if len_true > 0:
            sum_true_right_T1 += true_values_right_T1
            sum_true_right_T2 += true_values_right_T2
            sum_true_right_both += true_values_right_both
            len_of_true += len_true

        if len_false > 0:
            sum_false_right_T1 += false_values_right_T1
            sum_false_right_T2 += false_values_right_T2
            sum_false_right_both += false_values_right_both
            len_of_false += len_false

    return sum_true_right_T1, sum_false_right_T1, sum_true_right_T2, sum_false_right_T2, sum_true_right_both, sum_false_right_both, len_of_true, len_of_false


def mean_accuracy(nn, subjects, allFalse=False, into_file=False, threshold_for_T1=0.65, threshold_for_T2=0.65):
    sum_true_right_T1 = [0 for i in range(len(subjects))]
    sum_false_right_T1 = [0 for i in range(len(subjects))]
    sum_true_right_T2 = [0 for i in range(len(subjects))]
    sum_false_right_T2 = [0 for i in range(len(subjects))]
    sum_true_right_both = [0 for i in range(len(subjects))]
    sum_false_right_both = [0 for i in range(len(subjects))]
    len_of_true = [0 for i in range(len(subjects))]
    len_of_false = [0 for i in range(len(subjects))]


    for subject in range(len(subjects)):
        if allFalse:
            file = "predicted_data_allFalse_s" + str(subject) + ".h5"
        else:
            file = "predicted_data_s" + str(subject) + ".h5"
        sum_true_right_T1[subject], sum_false_right_T1[subject], sum_true_right_T2[subject], sum_false_right_T2[
            subject], sum_true_right_both[
            subject], sum_false_right_both[subject], len_of_true[subject], len_of_false[subject] = read_predicted_file(
            nn, subjects[subject],
            file=file,
            threshold_for_T1=threshold_for_T1,
            threshold_for_T2=threshold_for_T2)

    true_right_T1 = sum(sum_true_right_T1)
    false_right_T1 = sum(sum_false_right_T1)

    true_right_T2 = sum(sum_true_right_T2)
    false_right_T2 = sum(sum_false_right_T2)

    true_right_both = sum(sum_true_right_both)
    false_right_both = sum(sum_false_right_both)
    len_of_true_sum = sum(len_of_true)
    len_of_false_sum = sum(len_of_false)

    if into_file:
        aepath = home_repo + 'nn_' + str(nn) + '/' + global_task + '/'
        file = open(aepath + "mean_acc.txt", "w")
        file.write("Mean acuracy of subjects {} \n".format(list_of_subjects))
        if allFalse:
            file.write('For T1 type II error in cross-validation: {} \n '.format(
                (1 - false_right_T1 / len_of_false_sum)))

            file.write('For T2  type II error in cross-validation: {} \n '.format(
                (1 - false_right_T2 / len_of_false_sum)))

            file.write(
                'For both type II error in cross-validation: {} \n '.format(
                    (1 - false_right_both / len_of_false_sum)))

        else:
            file.write(
                'For T1:  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                    (1 - true_right_T1 / len_of_true_sum), (1 - false_right_T1 / len_of_false_sum)))

            file.write(
                'For T2  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                    (1 - true_right_T2 / len_of_true_sum), (1 - false_right_T2 / len_of_false_sum)))

            file.write(
                'For both  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                    (1 - true_right_both / len_of_true_sum), (1 - false_right_both / len_of_false_sum)))

            file.write(
                'For T1 accuracy overall: {} \n '.format(
                    (true_right_T1 + false_right_T1) / (len_of_true_sum + len_of_false_sum)))

            file.write(
                'For T2  accuracy overall {} \n '.format(
                    (true_right_T2 + false_right_T2) / (len_of_true_sum + len_of_false_sum)))

            file.write('For both  accuracy overall {} '.format(
                (true_right_both + false_right_both) / (len_of_true_sum + len_of_false_sum)))

        file.close()

    else:

        if allFalse:
            print("===================================================\n OVERALL ALL FALSE")

            print('For T1 type II error in cross-validation: {} \n '.format(
                (1 - false_right_T1 / len_of_false_sum)))

            print('For T2  type II error in cross-validation: {} \n '.format(
                (1 - false_right_T2 / len_of_false_sum)))

            print(
                'For both type II error in cross-validation: {} \n '.format(
                    (1 - false_right_both / len_of_false_sum)))

        else:
            print("===================================================\n OVERALL TEST DATA")

            print(
                'For T1:  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                    (1 - true_right_T1 / len_of_true_sum), (1 - false_right_T1 / len_of_false_sum)))

            print('For T2  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                (1 - true_right_T2 / len_of_true_sum), (1 - false_right_T2 / len_of_false_sum)))

            print(
                'For both  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                    (1 - true_right_both / len_of_true_sum), (1 - false_right_both / len_of_false_sum)))

            print(
                'For T1 accuracy overall: {} '.format(
                    (true_right_T1 + false_right_T1) / (len_of_true_sum + len_of_false_sum)))

            print(
                'For T2  accuracy overall {} '.format(
                    (true_right_T2 + false_right_T2) / (len_of_true_sum + len_of_false_sum)))

            print('For both  accuracy overall {} '.format(
                (true_right_both + false_right_both) / (len_of_true_sum + len_of_false_sum)))


if __name__ == '__main__':
    mean_accuracy("simple_1_with_dropout_2", [2,3,4], into_file=False, allFalse=False,
                  threshold_for_T1=0.5, threshold_for_T2=0.5)

    mean_accuracy("simple_1_with_dropout_2", [2,3,4], into_file=False, allFalse=True,
                  threshold_for_T1=0.5, threshold_for_T2=0.5)
