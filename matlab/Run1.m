clear all
%% create sub group
selected_channels={'Pz..','P3..','P4..','Cpz.','Cp3.','Cp4.','C2..','C4..','Fcz.','C5..','C6..','Fc3.','Fc4.','Fz..','F3..','F4..'};
task=1;%set number of task
%Combine_features(task,selected_channels);

%% PCA
size_of_vector=220; % size of input for SVM (use all true data and rest add with random falses)
%PCA_make_data_new( task, size_of_vector, selected_channels);


number_sub_channel=2; % how many subchannels left after PCA, convolute electrodes(chanels)
knn=0;
Size_of_feat=20;%number of feature, when stop grid selection
KernelSVM='rbf';%'rbf' or 'linear' optional
fast=1;%if 1 run without optimization
%mean_result_T1 = PCA_SVM_T1_new( task, Size_of_feat, KernelSVM,  fast, knn, selected_channels, number_sub_channel); %return mean accuracy
%mean_result_T2 = PCA_SVM_T2_new( task, Size_of_feat, KernelSVM,  fast, knn, selected_channels, number_sub_channel); %return mean accuracy
fast_check=1;%if 1 run without optimization
PCA_check( task, fast, fast_check, knn, selected_channels, number_sub_channel );


%% here Python should be run








