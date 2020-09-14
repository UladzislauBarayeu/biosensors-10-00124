# ================================================
# Author: Nastassya Horlava
# Github: @HorlavaNastassya
# Email: g.nasta.work@gmail.com
# ===============================================

import numpy as np

true_vector = np.array([1.0, 0.0])
false_vector = np.array([0.0, 1.0])
number_of_trials = 105
number_of_subjects = 105
general_repo = '/data/leuven/329/vsc32985/Vlad01/'
additional_folder_for_nn='../'
home_repo = general_repo + 'Data/neural_network_models/'
repo_with_raw_data = general_repo + 'Data/Result_json/'
matlab_repo_for_saving_svm = general_repo + 'Data/NN_convoluted/'
python_repo_for_saving_predicted = general_repo + 'Data/NN_predicted/'

earlystop = False
data_len = 210
two_times = True
batch_size = 140
lr = 0.001
epoch = 700
global_task = "Task1"

