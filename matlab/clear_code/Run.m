clear all
%% preproc
folderpath='Data/input/';
task=1;%set number of task
preproces( task, folderpath );

%% EMD
type=1;% 1-emd from lena 2-emd from internet 3-NAMEMD
Extract_EMD_feature(task, type);

%% Extract feature
freq_feature( task );
entropy_feature( task );
Combine_features(task);

%% PCA
number_sub_channel=2; % how many subchannels left after PCA, convolute electrodes(chanels)
PCA_get( task, number_sub_channel);
size_of_vector=225; % size of input for SVM (use all true data and rest add with random falses)
PCA_make_data( task,  size_of_vector);

Size_of_feat=10;%number of feature, when stop grid selection
KernelSVM='rbf';%'rbf' or 'linear' optional
fast=1;%if 1 run without optimization
mean_result_T1 = PCA_SVM_T1( task, Size_of_feat, KernelSVM,  fast); %return mean accuracy
mean_result_T2 = PCA_SVM_T2( task, Size_of_feat, KernelSVM,  fast); %return mean accuracy
fast_check=1;%if 1 run without optimization
PCA_check( task, fast, fast_check );


%% Run python
command = 'cd article/Python/nn_models/';
[status,cmdout] = system(command);
command = 'python create_nn211.py';
[status,cmdout] = system(command);
command = 'cd..';
[status,cmdout] = system(command);
command = 'python create_nn211.py';
[status,cmdout] = system(command);
command = 'cd..';
[status,cmdout] = system(command);
%% upgrade results from NN
List_of_subject={'s05','s15','s25','s35','s45','s55','s65','s75','s85','s95'}; % set list of subject
KernelSVM='linear';%'rbf' or 'linear' optional
fast=1;%if 1 run without optimization
nn=21;%set type of NN
Size_of_feat=10;%number of feature, when stop grid selection


mean_result_T1 = SVM_after_NN_T1( task, KernelSVM, List_of_subject, nn, Size_of_feat);%return mean accuracy
mean_result_T2 = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat);%return mean accuracy

fast_check=1;
NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM);  % check model



%%
fast=1;%if 1 run without optimization
nn=21;%set type of NN
List_of_subject={'s05','s15','s25','s35','s45','s55','s65','s75','s85','s95'}; % set list of subject
ROC_all( task, fast,  nn, List_of_subject);










