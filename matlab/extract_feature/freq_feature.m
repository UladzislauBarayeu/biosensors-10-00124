function [] = freq_feature( task )
%UNTITLED extract frequance features

size_of_subjects=105;

%%
for Number_of_subject=1:size_of_subjects
    
    name_file=strcat('Data/preproces/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    
    preProcesed=load(char(name_file));
    [T1_freq_feature,freqlabels]=...
        freq_feat(preProcesed.T1, preProcesed.channel_names, preProcesed.sample_rate);
    [T2_freq_feature,~]=...
        freq_feat(preProcesed.T2, preProcesed.channel_names, preProcesed.sample_rate);
    
    %T1_freq_feature=T1_freq_features{Number_of_subject};
    %T2_freq_feature=T2_freq_features{Number_of_subject};
    % Define the folder where to store the data
    outputDir = strcat('Data/Processed/freq_features/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
    % Write it to disk
    save(outputfilename,'T1_freq_feature','T2_freq_feature','freqlabels');
end



end

