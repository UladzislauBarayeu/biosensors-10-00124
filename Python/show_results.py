import os
import h5py
import numpy as np
from configurations import *
true_vector=np.array([1.0, 0.0])
false_vector=np.array([0.0, 1.0])
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def read_predicted_file(nn, s, file='predicted_data.h5', threshold_for_T1=0.5, threshold_for_T2=0.5):
    aepath = home_repo+'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'

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
        len_true = ((test_y == true_vector).sum()//2)
        len_false = ((test_y == false_vector).sum()//2)

        true_values_right_both = 0
        false_values_right_both = 0

        true_values_right_T1 = 0
        false_values_right_T1 = 0

        true_values_right_T2 = 0
        false_values_right_T2 = 0

        for j in range(len(test_y)):

            if (y_pred_1[j][0] >0.65):
                t1 = [1.0, 0.0]
            else:
                t1 = [0.0, 1.0]

            if (y_pred_2[j][0] >0.65):
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

def plot_Roc(y_test, y_predicted):
    # Compute ROC curve and ROC area for each class
    y_test=np.array([item for sublist in y_test for item in sublist])
    y_predicted=np.array([item for sublist in y_predicted for item in sublist])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_predicted[:, i], y_test[:, i], pos_label=1 )
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def mean_accuracy(nn, subjects, allFalse=False, into_file=False, threshold_for_T1=0.65, threshold_for_T2=0.65):

    sum_true_right_T1 = [0 for i in range(len(subjects))]
    sum_false_right_T1 = [0 for i in range(len(subjects))]
    sum_true_right_T2 = [0 for i in range(len(subjects))]
    sum_false_right_T2 = [0 for i in range(len(subjects))]
    sum_true_right_both = [0 for i in range(len(subjects))]
    sum_false_right_both = [0 for i in range(len(subjects))]
    len_of_true = [0 for i in range(len(subjects))]
    len_of_false = [0 for i in range(len(subjects))]

    for i in range(len(subjects)):
        sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i] = read_predicted_file(nn, subjects[i], threshold_for_T1=threshold_for_T1, threshold_for_T2=threshold_for_T2)

    true_right_T1 = sum(sum_true_right_T1)
    false_right_T1 = sum(sum_false_right_T1)

    true_right_T2 = sum(sum_true_right_T2)
    false_right_T2 = sum(sum_false_right_T2)

    true_right_both = sum(sum_true_right_both)
    false_right_both = sum(sum_false_right_both)
    len_of_true_sum = sum(len_of_true)
    len_of_false_sum = sum(len_of_false)

    if into_file:
        aepath = home_repo + 'two-task-nn/nn' + str(nn) + '/'
        file = open(aepath+"mean_acc.txt", "w")
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
            file.write('For T1:  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                (1 - true_right_T1 / len_of_true_sum), (1 - false_right_T1 / len_of_false_sum)))

            file.write('For T2  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                (1 - true_right_T2 / len_of_true_sum), (1 - false_right_T2 / len_of_false_sum)))

            file.write('For both  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n \n '.format(
                (1 - true_right_both / len_of_true_sum), (1 - false_right_both / len_of_false_sum)))

            file.write(
                'For T1 accuracy overall: {} \n '.format((true_right_T1 + false_right_T1) / (len_of_true_sum + len_of_false_sum)))

            file.write(
                'For T2  accuracy overall {} \n '.format((true_right_T2 + false_right_T2) / (len_of_true_sum + len_of_false_sum)))

            file.write('For both  accuracy overall {} '.format(
                (true_right_both + false_right_both) / (len_of_true_sum + len_of_false_sum)))

        file.close()

    else:
        print("===================================================\n OVERALL")

        if allFalse:
            print('For T1 type II error in cross-validation: {} \n '.format(
                (1 - false_right_T1 / len_of_false_sum)))

            print('For T2  type II error in cross-validation: {} \n '.format(
                (1 - false_right_T2 / len_of_false_sum)))

            print(
                'For both type II error in cross-validation: {} \n '.format(
                    (1 - false_right_both / len_of_false_sum)))

        else:
            print('For T1:  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                (1 - true_right_T1 / len_of_true_sum), (1 - false_right_T1 / len_of_false_sum)))

            print('For T2  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                (1 - true_right_T2 / len_of_true_sum), (1 - false_right_T2 / len_of_false_sum)))

            print('For both  \n type I error in cross-validation: {} \n type II error in cross-validation: {} \n '.format(
                (1 - true_right_both / len_of_true_sum), (1 - false_right_both / len_of_false_sum)))

            print(
                'For T1 accuracy overall: {} '.format((true_right_T1 + false_right_T1) / (len_of_true_sum + len_of_false_sum)))

            print(
                'For T2  accuracy overall {} '.format((true_right_T2 + false_right_T2) / (len_of_true_sum + len_of_false_sum)))

            print('For both  accuracy overall {} '.format(
                (true_right_both + false_right_both) / (len_of_true_sum + len_of_false_sum)))

if __name__ == '__main__':
    mean_accuracy(nn, list_of_subjects, into_file=True)
