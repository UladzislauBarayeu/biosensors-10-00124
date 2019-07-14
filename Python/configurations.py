import numpy as np

true_vector = np.array([1.0, 0.0])
false_vector = np.array([0.0, 1.0])
number_of_trials = 110
number_of_subjects = 105
general_repo = 'vsc-hard-mounts/leuven-user/329/vsc32985/'

additional_folder_for_nn=''
home_repo = general_repo + 'Data/neural_network_models/'
repo_with_raw_data = general_repo + 'Data/Result_json/'
matlab_repo_for_saving_svm = general_repo + 'Data/NN_convoluted/nn'
python_repo_for_saving_predicted = general_repo + 'Data/NN_predicted/nn'

earlystop = False
data_len = 220
two_times = True
batch_size = 140
lr = 0.001
epoch = 400

list_of_subjects = [i for i in range(1, number_of_subjects+1, 1)]
global_task = "Task1"

nn = "inception_1_with_small_kernel"
