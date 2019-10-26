clear all
%% upgrade results from NN
List_of_subject={};
% set list of subject which was compute by NN
for i=1:105
    List_of_subject=[List_of_subject num2str(i)];
end


knn=1;
KernelSVM='rbf';%'rbf' or 'linear' optional
fast=1;%if 1 run without optimization
nn='nnsimple_1_with_dropout_2';%set type of NN,'nnsimple_1_with_dropout_2','nninception_1_with_small_kernel'
Size_of_feat=20;%number of feature, when stop grid selection
task=1;
fast_check=1;threshold=0.5;

mean_result_T1 = SVM_after_NN_T1( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, knn);%return mean accuracy
mean_result_T2 = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, knn);%return mean accuracy


NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM, Size_of_feat, threshold);  % check model



%%
fast=1;%if 1 run without optimization
ROC_all( task, fast,  nn, List_of_subject);


