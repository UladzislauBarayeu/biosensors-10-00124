function [] = Combine_features( task )
%combine entropy and freq features

size_of_subjects=105;


for Number_of_subject=1:size_of_subjects
    
    name_file=strcat('Data/Processed/EMD_features/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    entropy_feat=load(char(name_file));
    
    name_file=strcat('Data/Processed/freq_features/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    freq_feat=load(char(name_file));
    
    Subject.result_label=entropy_feat.entropylabels;
    for nb_channel=1:size(freq_feat.freqlabels,1)
        for nb_feat=1:size(freq_feat.freqlabels,2)
            Subject.result_label{nb_channel,nb_feat+size(entropy_feat.entropylabels,2)}=freq_feat.freqlabels{nb_channel, nb_feat};
        end
    end
    
    for nb_trial=1:size(entropy_feat.T1_entropy_feature,2)
        trial=zeros(size(entropy_feat.T1_entropy_feature{nb_trial},1),...
            size(entropy_feat.T1_entropy_feature{nb_trial},2)+size(freq_feat.T1_freq_feature{nb_trial},2));
        for nb_channel=1:size(entropy_feat.T1_entropy_feature{nb_trial},1)
            trial(nb_channel,1:size(entropy_feat.T1_entropy_feature{nb_trial},2))=...
                entropy_feat.T1_entropy_feature{nb_trial}(nb_channel,:);
            trial(nb_channel,(size(entropy_feat.T1_entropy_feature{nb_trial},2)+1):(size(freq_feat.T1_freq_feature{nb_trial},2)+size(entropy_feat.T1_entropy_feature{nb_trial},2)))=...
                freq_feat.T1_freq_feature{nb_trial}(nb_channel,:);
        end 
        Subject.T1{nb_trial}=trial;
    end
    
    for nb_trial=1:size(entropy_feat.T2_entropy_feature,2)
        trial=zeros(size(entropy_feat.T2_entropy_feature{nb_trial},1),...
            size(entropy_feat.T2_entropy_feature{nb_trial},2)+size(freq_feat.T2_freq_feature{nb_trial},2));
        for nb_channel=1:size(entropy_feat.T2_entropy_feature{nb_trial},1)
            trial(nb_channel,1:size(entropy_feat.T2_entropy_feature{nb_trial},2))=...
                entropy_feat.T2_entropy_feature{nb_trial}(nb_channel,:);
            trial(nb_channel,(size(entropy_feat.T2_entropy_feature{nb_trial},2)+1):(size(freq_feat.T2_freq_feature{nb_trial},2)+size(entropy_feat.T2_entropy_feature{nb_trial},2)))=...
                freq_feat.T2_freq_feature{nb_trial}(nb_channel,:);
        end 
        Subject.T2{nb_trial}=trial;
    end
    
    %% Saving
    jsonStr = jsonencode(Subject); 
    outputDir = strcat('Data/Result_json/Task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    fid = fopen(strcat(outputDir, num2str(Number_of_subject),'.json'), 'w'); 
    if fid == -1, error('Cannot create JSON file'); end 
    fwrite(fid, jsonStr, 'char'); 
    fclose(fid);
    % Define the folder where to store the data
    outputDir = strcat('Data/Processed/Combined/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
    % Write it to disk
    save(outputfilename,'Subject');
end

end

