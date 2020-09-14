clear all
%% create sub group

channels_groups={...
     {'64_channels'},...
     {'F4..','Fp1.','Fp2.','C1..','C2..','Fc1.','Fc2.','F3..'},...
     {'F4..','Fp1.','Fp2.','C1..','C2..','Fc1.','Fc2.','F3..','Fz..','Fcz.','C3..','C4..','F1..','F2..','AF3.','AF4.'}...
    };

%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
size_of_sub=105;

for i=1:size(channels_groups,2)
    selected_channels=channels_groups{i};
    task=1;%set number of task
    Combine_features(task,selected_channels);

    %% PCA
    size_of_vector=210; % size of input for SVM (use all true data and rest add with random falses)
    PCA_make_data_new( task, size_of_vector, selected_channels, size_of_sub);

    number_sub_channel=2; % how many subchannels left after PCA, convolute electrodes(chanels)
    knn=0;
    
    Size_of_feat=10;%number of feature, when stop grid selection
    KernelSVM='rbf';%'rbf' or 'linear' optional
    fast=1;%if 1 run without optimization
    mean_result_T1 = PCA_SVM_T1_new( task, Size_of_feat, KernelSVM,  fast,...
        knn, selected_channels, number_sub_channel, size_of_sub); %return mean accuracy
    mean_result_T2 = PCA_SVM_T2_new( task, Size_of_feat, KernelSVM,  fast,...
        knn, selected_channels, number_sub_channel, size_of_sub); %return mean accuracy
    fast_check=1;%if 1 run without optimization
    threshold=0.60;
    PCA_check( task, fast, fast_check, knn, selected_channels, number_sub_channel, threshold, KernelSVM, size_of_sub );

end
%% here Python should be run








