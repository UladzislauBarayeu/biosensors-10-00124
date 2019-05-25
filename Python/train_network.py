from EEG_class import *
from network_utils import *
from sklearn.utils import shuffle
import os
from configurations import *

true_vector=np.array([1.0, 0.0])
false_vector=np.array([0.0, 1.0])

def train_both_tasks(nn, s, number_of_folds=5, number_for_test=10 ,epoch=160, period=2, lr=0.0001, two_times=False, batch_size=36, with_test=False):

    test1 = EEGdata()
    file = str(s) + '.json'
    test1.load_raw_data(file, directory=home_repo+'Task1/', task='T1', load_false_data_from_files=True,
                        data_len=number_for_test * number_of_folds)

    test2 = EEGdata()
    file = str(s) + '.json'
    test2.load_raw_data(file, directory=home_repo+'Task1/', task='T2', load_false_data_from_files=False, other=test1,
                        data_len=number_for_test * number_of_folds)

    train_data_T1, train_data_T2, labels = shuffle(np.copy(test1.all_data), np.copy(test2.all_data),
                                                 np.copy(test1.all_labels), random_state=0)

    file_raw = home_repo+'two-task-nn/nn' + str(nn) + '/test_conv_ae.json'

    aepath = home_repo+'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'
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

    if with_test:
        true_values_right_T1 = 0
        false_values_right_T1 = 0

        true_values_right_T2 = 0
        false_values_right_T2 = 0

        true_values_right_both = 0
        false_values_right_both = 0

        len_true_all=0
        len_false_all=0

    loss = 'mean_squared_error'
    trainning_acc_T1 = [[] for i in range(number_of_folds)]
    trainning_loss_T1 = [[] for i in range(number_of_folds)]
    trainning_acc_T2 = [[] for i in range(number_of_folds)]
    trainning_loss_T2 = [[] for i in range(number_of_folds)]

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
            info_T1_1 = train_autoencoder(h5file=file_raw, format="json",  batch_size=batch_size, optimizer=optimizer,
                                       train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                       loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr/10)
            info_T1_2 = train_autoencoder(h5file=file1, batch_size=batch_size, optimizer=optimizer,
                                                                      train_data=train_x_T1, labels=train_y, epoch=epoch//2, period=period,
                                                                      loss=loss, ae_name=file1)
            optimizer = optimizers.Adam(lr=lr)
            info_T2_1 = train_autoencoder(h5file=file_raw, format="json", batch_size=batch_size, optimizer=optimizer,
                                       train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
                                       loss=loss, ae_name=file2)
            optimizer = optimizers.Adam(lr=lr/10)
            info_T2_2= train_autoencoder(h5file=file2, batch_size=batch_size, optimizer=optimizer,
                                       train_data=train_x_T2, labels=train_y, epoch=epoch//2, period=period,
                                       loss=loss, ae_name=file2)

            trainning_acc_T1[i] = info_T1_1.history['acc']
            trainning_loss_T1[i] = info_T1_1.history['loss']
            trainning_acc_T2[i] = info_T2_1.history['acc']
            trainning_loss_T2[i] = info_T2_1.history['loss']

            trainning_acc_T1[i].extend(info_T1_2.history['acc'])
            trainning_loss_T1[i].extend(info_T1_2.history['loss'])
            trainning_acc_T2[i].extend(info_T2_2.history['acc'])
            trainning_loss_T2[i].extend(info_T2_2.history['loss'])

        else:
            optimizer = optimizers.Adam(lr=lr)
            info_T1 = train_autoencoder(h5file=file_raw,  format="json", batch_size=batch_size, optimizer=optimizer,
                                       train_data=train_x_T1, labels=train_y, epoch=epoch, period=period,
                                       loss=loss, ae_name=file1)
            info_T2 = train_autoencoder(h5file=file_raw, format="json", batch_size=batch_size, optimizer=optimizer,
                                       train_data=train_x_T2, labels=train_y, epoch=epoch, period=period,
                                       loss=loss, ae_name=file2)

            trainning_acc_T1[i] = info_T1.history['acc']
            trainning_loss_T1[i] = info_T1.history['loss']

            trainning_acc_T2[i] = info_T2.history['acc']
            trainning_loss_T2[i] = info_T2.history['loss']

        if with_test:

            network1 = load_network(file1)
            y_pred_1 = network1.predict(test_sample_T1_all_folds[i])

            network2 = load_network(file2)
            y_pred_2 = network2.predict(test_sample_T2_all_folds[i])

            test_y=test_y_all_folds[i]
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


                print('for T1 type I was predicted {} out of {}'.format(true_values_right_T1, len_true_all))
                print('for T2 type I was predicted {} out of {}'.format(true_values_right_T2, len_true_all))
                print('for both type I was predicted {} out of {}'.format(true_values_right_both, len_true_all))


                print('for T1 type II was predicted {} out of {}'.format(false_values_right_T1, len_false_all))
                print('for T2 type II was predicted {} out of {}'.format(false_values_right_T2, len_false_all))
                print('for both type II was predicted {} out of {}'.format(false_values_right_both, len_false_all))

    with h5py.File(aepath+'/train_results.h5', 'w') as f:
        for m in range(number_of_folds):
            f.create_dataset("trainning_acc_T1_"+str(m), data=np.array(trainning_acc_T1[m], dtype=np.float64))
            f.create_dataset("trainning_loss_T1_"+str(m), data=np.array(trainning_loss_T1[m], dtype=np.float64))
            f.create_dataset("trainning_acc_T2_" + str(m), data=np.array(trainning_acc_T2[m], dtype=np.float64))
            f.create_dataset("trainning_loss_T2_" + str(m), data=np.array(trainning_loss_T2[m], dtype=np.float64))

if __name__ == '__main__':
    train_both_tasks(211, 5, two_times=True, batch_size=140, lr=0.001, epoch=4, number_for_test=44)








