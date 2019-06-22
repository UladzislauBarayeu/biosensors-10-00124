from EEG_class import *
from network_utils import *
from sklearn.utils import shuffle
import os
from configurations import *


def train_both_tasks(nn, s, number_of_folds=5, number_for_test=10 ,epoch=160, period=2, lr=0.0001, two_times=False, batch_size=36, with_test=False, loss = 'mean_squared_error', global_task='Task1'):

    test1 = EEGdata()
    file = str(s) + '.json'
    test1.load_raw_data(file, global_task=global_task, task='T1', load_false_data_from_files=True,
                        data_len=number_for_test * number_of_folds)

    test2 = EEGdata()
    file = str(s) + '.json'
    test2.load_raw_data(file, global_task=global_task, task='T2', load_false_data_from_files=False, other=test1,
                        data_len=number_for_test * number_of_folds)

    train_data_T1, train_data_T2, labels = shuffle(np.copy(test1.all_data), np.copy(test2.all_data),
                                                 np.copy(test1.all_labels), random_state=0)

    file_raw = home_repo+'nn_' + str(nn) + '/test_conv_ae.h5'

    aepath = home_repo+'nn_' + str(nn) + '/' + str(s) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)

    train_sample_T1_all_folds = [[] for i in range(number_of_folds)]
    test_sample_T1_all_folds=[[] for i in range(number_of_folds)]
    train_sample_T2_all_folds = [[] for i in range(number_of_folds)]
    test_sample_T2_all_folds = [[] for i in range(number_of_folds)]
    train_y_all_folds=[[] for i in range(number_of_folds)]
    test_y_all_folds=[[] for i in range(number_of_folds)]
    minmax_T1_all_folds=[[] for i in range(number_of_folds)]
    minmax_T2_all_folds=[[] for i in range(number_of_folds)]


    for i in range(number_of_folds):
        test_x_T1= train_data_T1[i*number_for_test:i*number_for_test+number_for_test]
        X_T1=np.concatenate((train_data_T1[:i*number_for_test], train_data_T1[i*number_for_test+number_for_test:]), axis=0)
        train_x_T1, minmax_T1=normalize_data(X_T1)
        test_x_T1=normalize_data(test_x_T1, minmax_T1)

        test_x_T2 = train_data_T2[i * number_for_test:i * number_for_test + number_for_test]
        X_T2 = np.concatenate((train_data_T2[:i * number_for_test], train_data_T2[i * number_for_test + number_for_test:]),
                             axis=0)
        train_x_T2, minmax_T2 = normalize_data(X_T2)
        test_x_T2 = normalize_data(test_x_T2, minmax_T2)

        test_y = labels[i * number_for_test:i * number_for_test + number_for_test]
        train_y=np.concatenate((labels[:i*number_for_test], labels[i*number_for_test+number_for_test:]), axis=0)

        train_sample_T1_all_folds[i]=train_x_T1
        test_sample_T1_all_folds[i] = test_x_T1
        train_sample_T2_all_folds[i] = train_x_T2
        test_sample_T2_all_folds[i] = test_x_T2
        train_y_all_folds[i] = train_y
        test_y_all_folds[i] = test_y
        minmax_T1_all_folds[i] = minmax_T1
        minmax_T2_all_folds[i] = minmax_T2

    with h5py.File(aepath + 'data_for_training.h5', 'w') as f:
        d = f.create_dataset("train_sample_T1", data=np.array(train_sample_T1_all_folds, dtype=np.float64))
        d = f.create_dataset("test_sample_T1", data=np.array(test_sample_T1_all_folds, dtype=np.float64))

        d = f.create_dataset("train_sample_T2", data=np.array(train_sample_T2_all_folds, dtype=np.float64))
        d = f.create_dataset("test_sample_T2", data=np.array(test_sample_T2_all_folds, dtype=np.float64))

        d = f.create_dataset("train_labels", data=np.array(train_y_all_folds, dtype=np.float64))
        d = f.create_dataset("test_labels", data=np.array(test_y_all_folds, dtype=np.float64))

        d = f.create_dataset("minmax_T1", data=np.array(minmax_T1_all_folds, dtype=np.float64))
        d = f.create_dataset("minmax_T2", data=np.array(minmax_T2_all_folds, dtype=np.float64))

    for i in range (number_of_folds):
        train_x_T1 =train_sample_T1_all_folds[i]
        train_x_T2 = train_sample_T2_all_folds[i]
        train_y=train_y_all_folds[i]

        if not os.path.exists(aepath+'/T1/'):
            os.makedirs(aepath+'/T1/')
        if not os.path.exists(aepath+'/T2/'):
            os.makedirs(aepath+'/T2/')

        file1 = aepath+'T1/test_conv_ae_'+str(i)+'.h5'
        file2 = aepath+'T2/test_conv_ae_'+str(i)+'.h5'

        if two_times:
            optimizer = optimizers.Adam(lr=lr)
            info_T1_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file1)

            optimizer = optimizers.Adam(lr=lr/10)
            info_T1_2 = train_nn(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr)
            info_T2_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file2)

            optimizer = optimizers.Adam(lr=lr/10)
            info_T2_2= train_nn(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
                                loss=loss, ae_name=file2)

        else:
            optimizer = optimizers.Adam(lr=lr)
            info_T1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T1, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file1)
            info_T2 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T2, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file2)

        if with_test:
            test_within_fold(file1, file2, test_sample_T1_all_folds[i], test_sample_T2_all_folds[i], test_y_all_folds[i])





def train_both_tasks_from_fold(nn, s, n_fold, epoch=160, period=2, lr=0.0001, two_times=False, batch_size=36, loss = 'mean_squared_error'):

    file_raw = home_repo + 'nn_' + str(nn) + '/test_conv_ae.h5'
    aepath = home_repo + 'nn_' + str(nn) + '/' + str(s) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        train_sample_T1_all_folds = f["train_sample_T1"][:]
        train_sample_T2_all_folds = f["train_sample_T2"][:]
        train_y_all_folds = f["train_labels"][:]


    number_of_folds=train_sample_T1_all_folds.shape[0]
    for i in range (n_fold, number_of_folds, 1):
        train_x_T1 =train_sample_T1_all_folds[i]
        train_x_T2 = train_sample_T2_all_folds[i]
        train_y=train_y_all_folds[i]

        if not os.path.exists(aepath+'/T1/'):
            os.makedirs(aepath+'/T1/')
        if not os.path.exists(aepath+'/T2/'):
            os.makedirs(aepath+'/T2/')

        file1 = aepath+'T1/test_conv_ae_'+str(i)+'.h5'
        file2 = aepath+'T2/test_conv_ae_'+str(i)+'.h5'

        if two_times:
            optimizer = optimizers.Adam(lr=lr)
            info_T1_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr/10)
            info_T1_2 = train_nn(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr)
            info_T2_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
                                 loss=loss, ae_name=file2)
            optimizer = optimizers.Adam(lr=lr/10)
            info_T2_2= train_nn(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
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
    train_both_tasks("simple_1", 5, two_times=False, batch_size=140, lr=0.0001, epoch=500, number_of_folds=5,
                     number_for_test=44, with_test=True)











