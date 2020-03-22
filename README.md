# article
1) Download data via the link https://archive.physionet.org/pn4/eegmmidb/ and put this data into your_path/Data/input folder
2) run Matlab/Run1.m
3) run Python/nn_models/nn_inception_1_with_small_kernel.py or Python/nn_models/nn_simple_1_with_dropout_2.py to create a network (Inception-like or VGG-like respectively) and change in Python/configurations.py variable nn accordingly
4) run Python/Run_subjects.py script
