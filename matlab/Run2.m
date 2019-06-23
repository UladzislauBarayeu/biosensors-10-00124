clear all
%% upgrade results from NN
List_of_subject={'s35','s15','s25'}; % set list of subject
KernelSVM='linear';%'rbf' or 'linear' optional
fast=1;%if 1 run without optimization
nn=211;%set type of NN
Size_of_feat=10;%number of feature, when stop grid selection
task=1;

mean_result_T1 = SVM_after_NN_T1( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast);%return mean accuracy
mean_result_T2 = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast);%return mean accuracy

fast_check=1;
NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM);  % check model



%%
fast=1;%if 1 run without optimization
nn=21;%set type of NN
List_of_subject={'s05','s15','s25','s35','s45','s55','s65','s75','s85','s95'}; % set list of subject
ROC_all( task, fast,  nn, List_of_subject);


