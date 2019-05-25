%step 3: make EMD for each trial
%save all data like proccesed

%% set
clear all;
task=1;%set number of task
size_of_subjects=105;%set

%%
T1_EMD_par={};
T2_EMD_par={};
labels_par={};
number_of_IMFS_par={};
for Number_of_subject=1:size_of_subjects
    
    name_file=strcat('Data/preproces/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    
    preProcesed=load(char(name_file));
    [T1_EMD_par{Number_of_subject},labels_par{Number_of_subject},number_of_IMFS_par{Number_of_subject}]=...
        EMD(preProcesed.T1, preProcesed.channel_names);
    [T2_EMD_par{Number_of_subject},~,~]=...
        EMD(preProcesed.T2, preProcesed.channel_names);
    sample_rate_new_par{Number_of_subject}=preProcesed.sample_rate;
    
    T1_EMD=T1_EMD_par{Number_of_subject};
    T2_EMD=T2_EMD_par{Number_of_subject};
    labels=labels_par{Number_of_subject};
    sample_rate_new=sample_rate_new_par{Number_of_subject};
    number_of_IMFS=number_of_IMFS_par{Number_of_subject};
    fprintf('Saving the data...\n');
    % Define the folder where to store the data
    outputDir = strcat('Data/Processed/EMD/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
    % Write it to disk
    save(outputfilename,'T1_EMD','T2_EMD','labels','sample_rate_new','number_of_IMFS');
end

%%
%for Number_of_subject=1:size_of_subjects
%    T1_EMD=T1_EMD_par{Number_of_subject};
%    T2_EMD=T2_EMD_par{Number_of_subject};
%    labels=labels_par{Number_of_subject};
%    sample_rate_new=sample_rate_new_par{Number_of_subject};
%    number_of_IMFS=number_of_IMFS_par{Number_of_subject};
%    fprintf('Saving the data...\n');
%    % Define the folder where to store the data
%    outputDir = strcat('Processed/EMD/task',num2str(task),'/');
%    % Check if the folder exists , and if not, make it...
%    if ~exist(outputDir, 'dir')
%        mkdir(outputDir);
%    end
%    % Define the filename to store the data
%    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
%    % Write it to disk
%    save(outputfilename,'T1_EMD','T2_EMD','labels','sample_rate_new','number_of_IMFS');
%    
%end