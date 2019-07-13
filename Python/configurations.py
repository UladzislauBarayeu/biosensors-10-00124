import numpy as np
true_vector = np.array([1.0, 0.0])
false_vector = np.array([0.0, 1.0])
number_of_trials=110
number_of_subjects=105
home_repo = '../../Data/neural_network_models/'
repo_with_raw_data = '../../Data/Result_json/'
matlab_repo_for_saving_svm = '../../Data/NN_convoluted/nn'

earlystop = False
data_len=220
two_times=True
batch_size=140
lr=0.001
epoch=400

list_of_subjects = [1]
global_task = "Task1"

nn = "simple_1_with_dropout_2"
