from EEG_class import *
from network_utils import *
from sklearn.utils import shuffle
import os
from configurations import *
import h5py

def train_both_tasks(nn, subject, number_of_folds=5, data_len=200, epoch=160, period=2, lr=0.0001, two_times=False,
                     batch_size=36, loss='mean_squared_error', global_task='Task1', earlystop=False, channels='16'):
    test1 = EEGdata()
    test1.load_raw_data(subject, global_task=global_task, task='T1', load_false_data_from_files=True,
                        data_len=data_len, channels=channels)

    test2 = EEGdata()
    test2.load_raw_data(subject, global_task=global_task, task='T2', load_false_data_from_files=False, other=test1,
                        data_len=data_len)

    train_data_T1, train_data_T2, labels = shuffle(np.copy(test1.all_data), np.copy(test2.all_data),
                                                   np.copy(test1.all_labels), random_state=0)


    file_raw = home_repo +str(channels) + '/nn_' + str(nn) + '/test_conv_ae.h5'

    aepath = home_repo + str(channels) + '/nn_' + str(nn) + '/' +global_task+'/'+ str(subject) + '/'

    if not os.path.exists(aepath):
        os.makedirs(aepath)

    info_file = open(aepath + "train_params.txt", "w")
    info_file.write(
        "data len={}\n two_times={} \n batch_size={}\n lr={}\n epoch={}\n  earlystop={}\n loss = {}".format(data_len,
                                                                                                            two_times,
                                                                                                            batch_size,
                                                                                                            lr, epoch,
                                                                                                            earlystop,
                                                                                                            loss))
    info_file.close()

    train_sample_T1_all_folds = [[] for i in range(number_of_folds)]
    test_sample_T1_all_folds = [[] for i in range(number_of_folds)]
    train_sample_T2_all_folds = [[] for i in range(number_of_folds)]
    test_sample_T2_all_folds = [[] for i in range(number_of_folds)]
    train_y_all_folds = [[] for i in range(number_of_folds)]
    test_y_all_folds = [[] for i in range(number_of_folds)]
    minmax_T1_all_folds = [[] for i in range(number_of_folds)]
    minmax_T2_all_folds = [[] for i in range(number_of_folds)]

    number_for_test = data_len // number_of_folds

    for fold in range(number_of_folds):
        test_x_T1 = train_data_T1[fold * number_for_test:fold * number_for_test + number_for_test]
        X_T1 = np.concatenate(
            (train_data_T1[:fold * number_for_test], train_data_T1[fold * number_for_test + number_for_test:]), axis=0)
        train_x_T1, minmax_T1 = normalize_data(X_T1)
        test_x_T1 = normalize_data(test_x_T1, minmax_T1)

        test_x_T2 = train_data_T2[fold * number_for_test:fold * number_for_test + number_for_test]
        X_T2 = np.concatenate(
            (train_data_T2[:fold * number_for_test], train_data_T2[fold * number_for_test + number_for_test:]),
            axis=0)
        train_x_T2, minmax_T2 = normalize_data(X_T2)
        test_x_T2 = normalize_data(test_x_T2, minmax_T2)

        test_y = labels[fold * number_for_test:fold * number_for_test + number_for_test]
        train_y = np.concatenate((labels[:fold * number_for_test], labels[fold * number_for_test + number_for_test:]), axis=0)

        train_sample_T1_all_folds[fold] = train_x_T1
        test_sample_T1_all_folds[fold] = test_x_T1
        train_sample_T2_all_folds[fold] = train_x_T2
        test_sample_T2_all_folds[fold] = test_x_T2
        train_y_all_folds[fold] = train_y
        test_y_all_folds[fold] = test_y
        minmax_T1_all_folds[fold] = minmax_T1
        minmax_T2_all_folds[fold] = minmax_T2

    with h5py.File(aepath + 'data_for_training.h5', 'w') as f:
        d = f.create_dataset("train_sample_T1", data=np.array(train_sample_T1_all_folds, dtype=np.float64))
        d = f.create_dataset("test_sample_T1", data=np.array(test_sample_T1_all_folds, dtype=np.float64))

        d = f.create_dataset("train_sample_T2", data=np.array(train_sample_T2_all_folds, dtype=np.float64))
        d = f.create_dataset("test_sample_T2", data=np.array(test_sample_T2_all_folds, dtype=np.float64))

        d = f.create_dataset("train_labels", data=np.array(train_y_all_folds, dtype=np.float64))
        d = f.create_dataset("test_labels", data=np.array(test_y_all_folds, dtype=np.float64))

        d = f.create_dataset("minmax_T1", data=np.array(minmax_T1_all_folds, dtype=np.float64))
        d = f.create_dataset("minmax_T2", data=np.array(minmax_T2_all_folds, dtype=np.float64))

    for fold in range(number_of_folds):
        train_x_T1 = train_sample_T1_all_folds[fold]
        train_x_T2 = train_sample_T2_all_folds[fold]
        train_y = train_y_all_folds[fold]

        if not os.path.exists(aepath + '/T1/'):
            os.makedirs(aepath + '/T1/')
        if not os.path.exists(aepath + '/T2/'):
            os.makedirs(aepath + '/T2/')

        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'

        if two_times:
            optimizer = optimizers.Adam(lr=lr)
            print("training in {} fold, for T1 for the 1st time".format(fold))
            info_T1_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1, earlystop=earlystop)

            optimizer = optimizers.Adam(lr=lr / 10)
            print("training in {} fold, for T1 for the 2nd time".format(fold))
            info_T1_2 = train_nn(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T1, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file1, earlystop=earlystop)
            optimizer = optimizers.Adam(lr=lr)

            print("training in {} fold, for T1 for the 1st time".format(fold))
            info_T2_1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2, earlystop=earlystop)

            optimizer = optimizers.Adam(lr=lr / 10)
            print("training in {} fold, for T2 for the 2nd time".format(fold))
            info_T2_2 = train_nn(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                 train_data=train_x_T2, labels=train_y, epoch=epoch // 2, period=period,
                                 loss=loss, ae_name=file2, earlystop=earlystop)

        else:
            optimizer = optimizers.Adam(lr=lr)
            print("training in {} fold, for T1".format(fold))
            info_T1 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T1, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file1, earlystop=earlystop)
            print("training in {} fold, for T2".format(fold))
            info_T2 = train_nn(h5file=file_raw, batch_size=batch_size, optimizer=optimizer,
                               train_data=train_x_T2, labels=train_y, epoch=epoch, period=period,
                               loss=loss, ae_name=file2, earlystop=earlystop)


def train_both_tasks_from_fold(nn, s, n_fold, epoch=160, period=2, lr=0.0001, two_times=False, batch_size=36,
                               loss='mean_squared_error', channels=16):
    file_raw = home_repo +str(channels)+'/nn_' + str(nn) + '/test_conv_ae.h5'
    aepath = home_repo + str(channels)+'/nn_' + str(nn) + '/' +global_task+'/'+ str(s) + '/'
    if not os.path.exists(aepath):
        os.makedirs(aepath)
    with h5py.File(aepath + 'data_for_training.h5', 'r') as f:
        train_sample_T1_all_folds = f["train_sample_T1"][:]
        train_sample_T2_all_folds = f["train_sample_T2"][:]
        train_y_all_folds = f["train_labels"][:]

    number_of_folds = train_sample_T1_all_folds.shape[0]

    for fold in range(n_fold, number_of_folds, 1):
        train_x_T1 = train_sample_T1_all_folds[fold]
        train_x_T2 = train_sample_T2_all_folds[fold]
        train_y = train_y_all_folds[fold]

        if not os.path.exists(aepath + '/T1/'):
            os.makedirs(aepath + '/T1/')
        if not os.path.exists(aepath + '/T2/'):
            os.makedirs(aepath + '/T2/')

        file1 = aepath + 'T1/test_conv_ae_' + str(fold) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(fold) + '.h5'

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
    train_both_tasks("inception_1_16_channels", 2, data_len=220, two_times=False, batch_size=140, lr=0.001,
                     epoch=2, number_of_folds=5)
