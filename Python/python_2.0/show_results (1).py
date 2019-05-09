import os
import random
import matplotlib.pyplot as plt
import h5py
import numpy as np
from main import *
true_vector=np.array([1.0, 0.0])
false_vector=np.array([0.0, 1.0])
import json
from keras import *
from network_utils import normalize_data
def show_res(nn, s):

    directory = 'output-nn/nn'+str(nn)+'/'+str(s)+'/'
    h5file = 'train_results.h5'
    f = h5py.File(directory + h5file, 'r')
    trainning_acc_1=[]
    trainning_loss_1=[]

    trainning_acc_2 = []
    trainning_loss_2= []

    #print('type I accuracy in cross-validation: ',f.attrs['true_val_right_percent'])

    #print('type II accuracy in cross-validation: ', f.attrs['false_val_right_percent'])

    num_of_folds=f.attrs['num_of_folds']

    for j in range(num_of_folds):

        trainning_acc_1.append(f['trainning_acc_T1_'+str(j)][:])
        trainning_loss_1.append(f['trainning_loss_T1_'+str(j)][:])

        trainning_acc_2.append(f['trainning_acc_T2_' + str(j)][:])
        trainning_loss_2.append(f['trainning_loss_T2_' + str(j)][:])

    fig1 = plt.figure(1)
    for j in range(num_of_folds):
        plt.plot(trainning_acc_1[j])
    plt.title('trainning accuracy for T1 for subject {}'.format(s))
    fig1.savefig(directory + '/training_acc_T1.png')
    plt.show()

    fig1 = plt.figure(1)
    for j in range(num_of_folds):
        plt.plot(trainning_acc_2[j])
    plt.title('trainning accuracy for T2 for subject {}'.format(s))
    fig1.savefig(directory + '/training_acc_T2.png')
    plt.show()

    fig2 = plt.figure(2)
    for j in range(num_of_folds):
        plt.plot(trainning_loss_1[j])
    plt.title('trainning loss for T1 for subject {}'.format(s))
    fig2.savefig(directory +  '/training_loss_T1.png')
    plt.show()

    fig2 = plt.figure(2)
    for j in range(num_of_folds):
        plt.plot(trainning_loss_2[j])
    plt.title('trainning loss for T2 for subject {}'.format(s))
    fig2.savefig(directory + '/training_loss_T2.png')
    plt.show()

    fig3 = plt.figure(3)

    false_val_right=f['false_val_right'][:]
    true_val_right=f['true_val_right'][:]
    impossible_for_false=f['impossible_for_false'][:]
    acc_for_false=np.zeros(num_of_folds)
    sum_false_right=0
    sum_true_right=0

    len_true=0
    len_false=0

    for i in range(num_of_folds):
        sum_false_right+=false_val_right[i][0]
        sum_false_right += impossible_for_false[i][0]

        sum_true_right+=true_val_right[i][0]

        len_false+=false_val_right[i][1]
        len_true+=true_val_right[i][1]
        print('type II was predicted right {} with {} impossible out of {}'.format(false_val_right[i][0],impossible_for_false[i][0],false_val_right[i][1]))
        acc_for_false[i]=(false_val_right[i][0]+impossible_for_false[i][0])/false_val_right[i][1]
    plt.title(
        'accuracy for true subjects while cross-validation \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n ,for subject {}'.format(
            sum_true_right / len_true, sum_false_right / len_false, s))
    plt.hist(acc_for_false)
    fig3.savefig(directory +  '/accuracy_for_false.png')
    plt.show()

    fig3 = plt.figure(3)
    plt.title('accuracy for true subjects while cross-validation \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n ,for subject {}'.format(
            sum_true_right/len_true, sum_false_right/len_false, s))
    true_val_right = f['true_val_right'][:]
    acc_for_false = np.zeros(num_of_folds)
    for i in range(num_of_folds):
        acc_for_false[i] = true_val_right[i][0] / true_val_right[i][1]
        print('type I was predicted right {} out of {}'.format(true_val_right[i][0],true_val_right[i][1]))

    plt.hist(acc_for_false)
    fig3.savefig(directory + '/accuracy_for_true.png')
    plt.show()
    f.close()

def show_res_one_task(nn, s):

    directory = 'output-nn/nn'+str(nn)+'/'+str(s)+'/'
    h5file = 'train_results.h5'
    f = h5py.File(directory + h5file, 'r')
    trainning_acc_1=[]
    trainning_loss_1=[]


    #print('type I accuracy in cross-validation: ',f.attrs['true_val_right_percent'])

    #print('type II accuracy in cross-validation: ', f.attrs['false_val_right_percent'])

    num_of_folds=f.attrs['num_of_folds']

    for j in range(num_of_folds):

        trainning_acc_1.append(f['trainning_acc_T1_'+str(j)][:])
        trainning_loss_1.append(f['trainning_loss_T1_'+str(j)][:])


    fig1 = plt.figure(1)
    for j in range(num_of_folds):
        plt.plot(trainning_acc_1[j])
    plt.title('trainning accuracy for T1 for subject {}'.format(s))
    fig1.savefig(directory + '/training_acc_T1.png')
    plt.show()



    fig2 = plt.figure(2)
    for j in range(num_of_folds):
        plt.plot(trainning_loss_1[j])
    plt.title('trainning loss for T1 for subject {}'.format(s))
    fig2.savefig(directory +  '/training_loss_T1.png')
    plt.show()


    fig3 = plt.figure(3)

    false_val_right=f['false_val_right'][:]
    true_val_right=f['true_val_right'][:]
    impossible_for_false=f['impossible_for_false'][:]
    acc_for_false=np.zeros(num_of_folds)
    sum_false_right=0
    sum_true_right=0

    len_true=0
    len_false=0

    for i in range(num_of_folds):
        sum_false_right+=false_val_right[i][0]
        sum_false_right += impossible_for_false[i][0]

        sum_true_right+=true_val_right[i][0]

        len_false+=false_val_right[i][1]
        len_true+=true_val_right[i][1]
        print('type II was predicted right {} with {} impossible out of {}'.format(false_val_right[i][0],impossible_for_false[i][0],false_val_right[i][1]))
        acc_for_false[i]=(false_val_right[i][0]+impossible_for_false[i][0])/false_val_right[i][1]
    plt.title(
        'accuracy for true subjects while cross-validation \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n ,for subject {}'.format(
            sum_true_right / len_true, sum_false_right / len_false, s))
    plt.hist(acc_for_false)
    fig3.savefig(directory +  '/accuracy_for_false.png')
    plt.show()

    fig3 = plt.figure(3)
    plt.title('accuracy for true subjects while cross-validation \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n ,for subject {}'.format(
            sum_true_right/len_true, sum_false_right/len_false, s))
    true_val_right = f['true_val_right'][:]
    acc_for_false = np.zeros(num_of_folds)
    for i in range(num_of_folds):
        acc_for_false[i] = true_val_right[i][0] / true_val_right[i][1]
        print('type I was predicted right {} out of {}'.format(true_val_right[i][0],true_val_right[i][1]))

    plt.hist(acc_for_false)
    fig3.savefig(directory + '/accuracy_for_true.png')
    plt.show()
    f.close()


def retest_accuracy(nn, s, number_of_folds=5):
    first_folder = 'two-task-nn'
    aepath = first_folder + '/nn' + str(nn) + '/' + str(s) + '/'

    outputFolder = aepath + '/data_cross_validation'
    acc_for_true_both = []
    acc_for_false_both = []

    acc_for_true_T1 = []
    acc_for_false_T1 = []

    acc_for_true_T2 = []
    acc_for_false_T2 = []

    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i + 1) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i + 1) + '.h5'
        with h5py.File(outputFolder + '/data_for_training_'+str(i)+'.h5', 'r') as f:

            test_x_1 = f["test_sample_T1"][:]
            test_x_2 = f["test_sample_T2"][:]
            test_y = f["test_labels"][:]

        network1 = load_network(file1)
        y_pred_1 = network1.predict(test_x_1)

        network2 = load_network(file2)
        y_pred_2 = network2.predict(test_x_2)

        len_true = ((test_y == true_vector).sum() // 2)
        len_false = ((test_y == false_vector).sum() // 2)

        true_values_right_both = 0
        false_values_right_both = 0

        true_values_right_T1 = 0
        false_values_right_T1 = 0

        true_values_right_T2 = 0
        false_values_right_T2 = 0

        for j in range(len(test_y)):

            if (y_pred_1[j][0] > 0.65):
                t1 = [1.0, 0.0]
            else:
                t1 = [0.0, 1.0]

            if (y_pred_2[j][0] > 0.65):
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
            print('for T1 type I was predicted {} out of {}'.format(true_values_right_T1, len_true))
            print('for T2 type I was predicted {} out of {}'.format(true_values_right_T2, len_true))
            print('for both type I was predicted {} out of {}'.format(true_values_right_both, len_true))
            acc_for_true_both.append([true_values_right_both, len_true])
            acc_for_true_T1.append([true_values_right_T1, len_true])
            acc_for_true_T2.append([true_values_right_T2, len_true])

        if len_false > 0:
            print('for T1 type II was predicted {} out of {}'.format(false_values_right_T1, len_false))
            print('for T2 type II was predicted {} out of {}'.format(false_values_right_T2, len_false))
            print('for both type II was predicted {} out of {}'.format(false_values_right_both, len_false))
            acc_for_false_T1.append([false_values_right_T1, len_false])
            acc_for_false_T2.append([false_values_right_T2, len_false])
            acc_for_false_both.append([false_values_right_both, len_false])

    sum_false_right_both = 0
    sum_true_right_both = 0
    sum_false_right_T1 = 0
    sum_true_right_T1 = 0
    sum_false_right_T2 = 0
    sum_true_right_T2 = 0
    len_of_false = 0
    len_of_true = 0

    for i in range(number_of_folds):
        sum_false_right_T1 += acc_for_false_T1[i][0]
        sum_true_right_T1 += acc_for_true_T1[i][0]

        sum_false_right_T2 += acc_for_false_T2[i][0]
        sum_true_right_T2 += acc_for_true_T2[i][0]

        sum_false_right_both += acc_for_false_both[i][0]
        sum_true_right_both += acc_for_true_both[i][0]

        len_of_false += acc_for_false_both[i][1]
        len_of_true += acc_for_true_both[i][1]

    print('For T1:  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        sum_true_right_T1 / len_of_true, sum_false_right_T1 / len_of_false))

    print('For T2  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        sum_true_right_T2 / len_of_true, sum_false_right_T2 / len_of_false))

    print('For both  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        sum_true_right_both / len_of_true, sum_false_right_both / len_of_false))
    return sum_true_right_T1,  sum_false_right_T1, sum_true_right_T2 ,  sum_false_right_T2, sum_true_right_both, sum_false_right_both, len_of_true, len_of_false

def predict_two_tasks(nn, s, number_of_folds=5):

    aepath = 'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'
    outputFolder = aepath + '/data_cross_validation'

    t1_test_data=[0 for i in range(number_of_folds)]
    t2_test_data=[0 for i in range(number_of_folds)]
    test_y= [0 for i in range(number_of_folds)]

    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i + 1) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i + 1) + '.h5'

        with h5py.File(outputFolder + '/data_for_training_'+str(i)+'.h5', 'r') as f:

            test_x_1 = f["test_sample_T1"][:]
            test_x_2 = f["test_sample_T2"][:]
            test_labels = f["test_labels"][:]

        model1 = load_network(file1)
        model2 = load_network(file2)

        test_y[i]=test_labels.tolist()

        test_data_1 = model1.predict(test_x_1)
        test_data_2 = model2.predict(test_x_2)

        t1_test_data[i] = test_data_1.tolist()
        t2_test_data[i] = test_data_2.tolist()



    jsondic = {'T1':t1_test_data,
               'T2': t2_test_data,
            'test_y': test_y}


    outfile = open(aepath+'predicted_data'+str(s)+'.txt', 'w')
    json.dump(jsondic, outfile)
    outfile.close()

def predict_all_false(nn, s, number_of_folds=5):
    aepath = 'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'
    outputFolder = aepath + '/data_cross_validation'
    all_predicted_data_T1 = [0 for i in range(number_of_folds)]
    all_predicted_data_T2 = [0 for i in range(number_of_folds)]

    all_conv_data_T1 = [0 for i in range(number_of_folds)]
    all_conv_data_T2 = [0 for i in range(number_of_folds)]

    all_T1=[]
    all_T2=[]
    for j in range(1, 106, 1):
        if j!=s:
            directory = 'Task1/'
            h5file = str(s) + '.json'
            path = directory + h5file
            json_data = open(path)
            d = json.load(json_data)
            json_data.close()
            data_T1 = np.array(d['Subject_old']['T1'][:22])
            data_T1 = np.transpose(data_T1, (0, 2, 1))
            data_T1=np.expand_dims(data_T1, axis=3)
            all_T1.extend(data_T1)
            data_T2 = np.array(d['Subject_old']['T2'][:22])
            data_T2 = np.transpose(data_T2, (0, 2, 1))
            data_T2 = np.expand_dims(data_T2, axis=3)
            all_T2.extend(data_T2)
    all_T1=np.array(all_T1)
    all_T2=np.array(all_T2)

    for i in range(number_of_folds):
        file1 = aepath + 'T1/test_conv_ae_' + str(i + 1) + '.h5'
        file2 = aepath + 'T2/test_conv_ae_' + str(i + 1) + '.h5'
        with h5py.File(outputFolder + '/data_for_training_'+str(i)+'.h5', 'r') as f:
            minmax_1 = f["minmax_T1"][:]
            minmax_2 = f["minmax_T2"][:]

        norm_T1=normalize_data(all_T1, minmax_1)
        norm_T2=normalize_data(all_T2, minmax_2)

        model1 = load_network(file1)
        model2 = load_network(file2)

        predicted_data_T1 = model1.predict(norm_T1)
        predicted_data_T2 = model2.predict(norm_T2)
        all_predicted_data_T1[i]=predicted_data_T1.tolist()
        all_predicted_data_T2[i]=predicted_data_T2.tolist()


        layer_name = 'flatten_2'
        intermediate_layer_model1 = Model(inputs=model1.input,
                                          outputs=model1.get_layer(layer_name).output)
        intermediate_layer_model2 = Model(inputs=model2.input,
                                          outputs=model2.get_layer(layer_name).output)

        conv_data_T1 = intermediate_layer_model1.predict(norm_T1)
        conv_data_T2 = intermediate_layer_model2.predict(norm_T2)

        all_conv_data_T1[i]=conv_data_T1.tolist()
        all_conv_data_T2[i]=conv_data_T2.tolist()


    jsondic_for_predicted = {'T1_all_false': all_predicted_data_T1,
               'T2_all_false': all_predicted_data_T2}

    outfile = open(aepath + 'predicted_data_all_false_subjects' + str(s) + '.txt', 'w')
    json.dump(jsondic_for_predicted, outfile)
    outfile.close()

    jsondic_for_conv = {'T1_all_false': all_predicted_data_T1,
                             'T2_all_false': all_predicted_data_T2}

    outfile = open(aepath + 'conv_data_all_false_subjects' + str(s) + '.txt', 'w')
    json.dump(jsondic_for_conv, outfile)
    outfile.close()




def read_predicted_file(nn, s, file='predicted_data', all_false=False):
    aepath = 'two-task-nn/nn' + str(nn) + '/' + str(s) + '/'

    sum_false_right_both = 0
    sum_true_right_both = 0
    sum_false_right_T1 = 0
    sum_true_right_T1 = 0
    sum_false_right_T2 = 0
    sum_true_right_T2 = 0
    len_of_false = 0
    len_of_true = 0


    json_data = open(aepath + file + str(s) + '.json')
    d = json.load(json_data)
    json_data.close()
    if all_false is False:
        t1_test_data =np.array(d['T1'])
        t2_test_data = np.array(d['T2'])
        test_y_all=np.array(d['test_y'])
    else:
        t1_test_data = np.array(d['T1_all_false'])
        t2_test_data = np.array(d['T2_all_false'])
        test_y_all=[[false_vector for i in range(t1_test_data.shape[1])] for j in range(t1_test_data.shape[0])]
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

            if (y_pred_1[j][0] > 0.61):
                t1 = [1.0, 0.0]
            else:
                t1 = [0.0, 1.0]

            if (y_pred_2[j][0] > 0.75):
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


    if all_false is False:
        # print('For T1:  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        #     sum_true_right_T1 / len_of_true, sum_false_right_T1 / len_of_false))
        # print('For T2  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        #     sum_true_right_T2 / len_of_true, sum_false_right_T2 / len_of_false))
        # print('For both  \n type I accuracy in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        #     sum_true_right_both / len_of_true, sum_false_right_both / len_of_false))
        return sum_true_right_T1, sum_false_right_T1, sum_true_right_T2, sum_false_right_T2, sum_true_right_both, sum_false_right_both, len_of_true, len_of_false
    else:
        return sum_false_right_T1, sum_false_right_T2, sum_false_right_both, len_of_false

if __name__=='__main__':
    sum_false_right_T1 = [0 for i in range(6)]
    sum_false_right_T2 = [0 for i in range(6)]
    sum_false_right_both = [0 for i in range(6)]
    len_of_false = [0 for i in range(6)]

    i = 0
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 4, 'predicted_data_all_false_subjects', True)
    i += 1
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 5, 'predicted_data_all_false_subjects', True)
    i += 1
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 6, 'predicted_data_all_false_subjects', True)
    i += 1
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 16, 'predicted_data_all_false_subjects', True)
    i += 1
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 17,'predicted_data_all_false_subjects', True)
    i += 1
    sum_false_right_T1[i], sum_false_right_T2[i], sum_false_right_both[i], len_of_false[i] = read_predicted_file(2, 18, 'predicted_data_all_false_subjects', True)
    false_right_T1 = sum(sum_false_right_T1)
    false_right_T2 = sum(sum_false_right_T2)
    false_right_both = sum(sum_false_right_both)
    len_of_false_sum = sum(len_of_false)
    print("===================================================\n OVERALL")
    print('For T1:  \n  type II error in cross-validation: {} \n '.format(1 - false_right_T1 / len_of_false_sum))

    print('For T2   \n type II error in cross-validation: {} \n '.format(1 - false_right_T2 / len_of_false_sum))

    print('For both \n type II error in cross-validation: {} \n '.format(1 - false_right_both / len_of_false_sum))

    sum_true_right_T1=[0 for i in range(6)]
    sum_false_right_T1=[0 for i in range(6)]
    sum_true_right_T2 =[0 for i in range(6)]
    sum_false_right_T2=[0 for i in range(6)]
    sum_true_right_both=[0 for i in range(6)]
    sum_false_right_both=[0 for i in range(6)]
    len_of_true=[0 for i in range(6)]
    len_of_false=[0 for i in range(6)]

    i=0
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 4)
    i+=1
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 5)
    i+=1
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 6)
    i+=1
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 16)
    i+=1
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 17)
    i+=1
    sum_true_right_T1[i], sum_false_right_T1[i], sum_true_right_T2[i], sum_false_right_T2[i], sum_true_right_both[i], sum_false_right_both[i], len_of_true[i], len_of_false[i]=read_predicted_file(2, 18)


    true_right_T1=sum(sum_true_right_T1)
    false_right_T1=sum(sum_false_right_T1)

    true_right_T2=sum(sum_true_right_T2)
    false_right_T2=sum(sum_false_right_T2)


    true_right_both=sum(sum_true_right_both)
    false_right_both=sum(sum_false_right_both)
    len_of_true_sum=sum(len_of_true)
    len_of_false_sum=sum(len_of_false)
    print("===================================================\n OVERALL")
    print('For T1:  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
       (1-true_right_T1 / len_of_true_sum), (1-false_right_T1 / len_of_false_sum)))

    print('For T2  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        (1-true_right_T2 / len_of_true_sum), (1-false_right_T2 / len_of_false_sum)))

    print('For both  \n type I error in cross-validation: {} \n type II accuracy in cross-validation: {} \n '.format(
        (1-true_right_both / len_of_true_sum), (1-false_right_both / len_of_false_sum)))

    print('For T1 accuracy overall: {} '.format((true_right_T1+false_right_T1) / (len_of_true_sum+len_of_false_sum)))

    print('For T2  accuracy overall {} '.format((true_right_T2+false_right_T2)/(len_of_true_sum+len_of_false_sum)))

    print('For both  accuracy overall {} '.format((true_right_both+false_right_both)/ (len_of_true_sum+len_of_false_sum)))