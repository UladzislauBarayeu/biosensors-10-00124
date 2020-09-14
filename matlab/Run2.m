%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
clear all
%% upgrade results from NN
List_of_subject={};
% set list of subject which was compute by NN
for i=1:105
    List_of_subject=[List_of_subject num2str(i)];
end



types={'64_channels','8_channels','16_channels'};

for i=1:length(types)
    KernelSVM='rbf';%'rbf' or 'linear' optional
    fast=1;%if 1 run without optimization
    channel_type=types{i};
    %nn_inception_1_
    %nn_simple_1_
    nn=strcat('nn_simple_1_',types{i});%set type of NN,'nnsimple_1_with_dropout_2','nninception_1_with_small_kernel'
    Size_of_feat=10;%number of feature, when stop grid selection
    task=1;
    fast_check=1;threshold=0.62;

    mean_result_T1 = SVM_after_NN_T1( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, channel_type);%return mean accuracy
    mean_result_T2 = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, channel_type);%return mean accuracy


    NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM, threshold, channel_type);  % check model
end

for i=1:length(types)
    KernelSVM='rbf';%'rbf' or 'linear' optional
    fast=1;%if 1 run without optimization
    channel_type=types{i};
    %nn_inception_1_
    %nn_simple_1_
    nn=strcat('nn_inception_1_',types{i});%set type of NN,'nnsimple_1_with_dropout_2','nninception_1_with_small_kernel'
    Size_of_feat=10;%number of feature, when stop grid selection
    task=1;
    fast_check=1;threshold=0.62;

    mean_result_T1 = SVM_after_NN_T1( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, channel_type);%return mean accuracy
    mean_result_T2 = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, channel_type);%return mean accuracy


    NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM, threshold, channel_type);  % check model
end
    

