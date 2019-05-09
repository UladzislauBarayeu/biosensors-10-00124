from train_network import *
#from show_results import *
train_both_tasks(2, 35, two_times=True, batch_size=36, lr=0.001, epoch=180)

# from export_for_matlab import  *
# export_nn_for_svm_two_tasks(21,25)