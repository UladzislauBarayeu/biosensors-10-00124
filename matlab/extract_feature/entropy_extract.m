%% set
clear all;
task=2;%set number of task
size_of_subjects=105;%set

%%
for Number_of_subject=1:size_of_subjects
    
    name_file=strcat('Data/Processed/EMD/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    
    Procesed_EMD=load(char(name_file));
    [ T1_entropy_feature, entropylabels ] = Extract_entropy_feat( Procesed_EMD.T1_EMD, Procesed_EMD.labels);
    [ T2_entropy_feature, ~ ] = Extract_entropy_feat( Procesed_EMD.T2_EMD, Procesed_EMD.labels);
    
    fprintf('Saving the data...\n');
    % Define the folder where to store the data
    outputDir = strcat('Data/Processed/EMD_features/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
    % Write it to disk
    save(outputfilename,'T1_entropy_feature','T2_entropy_feature','entropylabels');
end