from EEG_class import *
from network_utils import *
from sklearn.utils import shuffle
import os
from configurations import *


def train_both_tasks(nn, s, epoch=160, period=2, lr=0.0001, two_times=False,
                     batch_size=36, with_test=False, loss='mean_squared_error', global_task='Task1', earlystop=False):

    file_raw = home_repo + 'nn_' + str(nn) + '/test_conv_ae.h5'
    aepath = home_repo + 'nn_' + str(nn) + '/' + str(s) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)

    train_sample_T1_all_folds, train_sample_T2_all_folds, train_y_all_folds = load_train_data(subject=s, global_task='Task1')
    number_of_folds=train_sample_T1_all_folds.shape[0]

    for i in range(number_of_folds):
        train_x_T1 = train_sample_T1_all_folds[i]
        train_x_T2 = train_sample_T2_all_folds[i]
        train_y = train_y_all_folds[i]

        if not os.path.exists(aepath + '/T1/'):
            os.makedirs(aepath + '/T1/')
        if not os.path.exists(aepath + '/T2/'):
            os.makedirs(aepath + '/T2/')

        file1 = aepath + 'T1/test_conv_ae_' + str(i) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i) + '.h5'

        if two_times:
            optimizer = optimizers.Adam(lr=lr)
            info_T1_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1)

            optimizer = optimizers.Adam(lr=lr / 10)
            info_T1_2 = train_nn(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1)

            optimizer = optimizers.Adam(lr=lr)
            info_T2_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2)

            optimizer = optimizers.Adam(lr=lr / 10)
            info_T2_2 = train_nn(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2)

        else:
            optimizer = optimizers.Adam(lr=lr)
            info_T1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T1, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file1, earlystop=earlystop)
            info_T2 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T2, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file2, earlystop=earlystop)

        if with_test:

            test_within_fold(s, global_task, file1, file2, i)
            test_within_fold_allFalse(s, global_task, file1, file2, i)


def train_both_tasks_from_fold(nn, s, n_fold, epoch=160, period=2, lr=0.0001, two_times=False, batch_size=36,
                               loss='mean_squared_error'):
    file_raw = home_repo + 'nn_' + str(nn) + '/test_conv_ae.h5'
    aepath = home_repo + 'nn_' + str(nn) + '/' + str(s) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        train_sample_T1_all_folds = f["train_sample_T1"][:]
        train_sample_T2_all_folds = f["train_sample_T2"][:]
        train_y_all_folds = f["train_labels"][:]

    number_of_folds = train_sample_T1_all_folds.shape[0]
    for i in range(n_fold, number_of_folds, 1):
        train_x_T1 = train_sample_T1_all_folds[i]
        train_x_T2 = train_sample_T2_all_folds[i]
        train_y = train_y_all_folds[i]

        if not os.path.exists(aepath + '/T1/'):
            os.makedirs(aepath + '/T1/')
        if not os.path.exists(aepath + '/T2/'):
            os.makedirs(aepath + '/T2/')

        file1 = aepath + 'T1/test_conv_ae_' + str(i) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i) + '.h5'

        if two_times:
            optimizer = optimizers.Adam(lr=lr)
            info_T1_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr / 10)
            info_T1_2 = train_nn(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr)
            info_T2_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2)
            optimizer = optimizers.Adam(lr=lr / 10)
            info_T2_2 = train_nn(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2)

        else:
            optimizer = optimizers.Adam(lr=lr)
            info_T1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T1, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file1)
            info_T2 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T2, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file2)


if __name__ == '__main__':
    train_both_tasks('inception_1_with_small_kernel', 5, two_times=True, batch_size=140, lr=0.001, epoch=180, with_test=False, earlystop=False)

