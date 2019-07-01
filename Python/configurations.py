import numpy as np
true_vector=np.array([1.0, 0.0])
false_vector=np.array([0.0, 1.0])

home_repo='../../Data/neural_network_models/'
repo_with_raw_data='../../Data/Result_json_cut/'
matlab_repo_for_saving_svm='../../Data/NN_convoluted/nn'
matlab_repo_for_saving_all_false_svm='../../Data/Python_res/NN_test/nn'

list_of_subjects=[5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
global_task="Task1"

nn="inception_1_with_small_kernel"
