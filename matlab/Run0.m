%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================


%% preproc
folderpath='Data/input/';
task=1;%set number of task
min_trials=105;% min number of  trials in all subjects
preproces( task, folderpath, min_trials);% run only on matlab2016


%% EMD
type=0;% 1-emd from lena 2-emd from internet 3-NAMEMD, 0-original func
number_of_IMFS=4;%set how many IMF left
Extract_EMD_feature(task, type, number_of_IMFS);
% 
% %% Extract feature
freq_feature(task);
entropy_feature(task);