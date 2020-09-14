function [] = Combine_features( task, selected_channels )
%combine entropy and freq features
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
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
    %% select good
    good_index=[];
    if strcmp(selected_channels{1},'64_channels')
        for nb_channel=1:size(freq_feat.freqlabels,1)
            good_index=[good_index nb_channel];
        end
    else
        for nb_channel=1:size(freq_feat.freqlabels,1)
            for nb_channel_good=1:size(selected_channels,2)
                if strcmp(entropy_feat.entropylabels{nb_channel,1}(1:4),selected_channels{nb_channel_good})
                    good_index=[good_index nb_channel];
                end
            end
        end
    end
    
    Subject_new.result_label=cell(size(good_index,2),size(Subject.result_label,2));
    for nb_channel=1:size( Subject_new.result_label,1)
        for nb_feat=1:size( Subject_new.result_label,2)
             Subject_new.result_label{nb_channel,nb_feat}=Subject.result_label{good_index(nb_channel),nb_feat};
        end
    end
    
    for trial=1:size(Subject.T1,2)
        for nb_channel=1:size( Subject_new.result_label,1)
            for nb_feat=1:size( Subject_new.result_label,2)
                 Subject_new.T1{trial}(nb_channel,nb_feat)=Subject.T1{trial}(good_index(nb_channel),nb_feat);
            end
        end
    end
    for trial=1:size(Subject.T2,2)
        for nb_channel=1:size( Subject_new.result_label,1)
            for nb_feat=1:size( Subject_new.result_label,2)
                 Subject_new.T2{trial}(nb_channel,nb_feat)=Subject.T2{trial}(good_index(nb_channel),nb_feat);
            end
        end
    end
    clear Subject
    Subject=Subject_new;
    clear Subject_new
    name_folder=strcat(num2str(length(good_index)),'_channels');

    %% Saving
    jsonStr = jsonencode(Subject); 
    outputDir = strcat('Data/Result_json/Task',num2str(task),'/',name_folder,'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    fid = fopen(strcat(outputDir, num2str(Number_of_subject),'.json'), 'w'); 
    if fid == -1, error('Cannot create JSON file'); end 
    fwrite(fid, jsonStr, 'char'); 
    fclose(fid);
    % Define the folder where to store the data
    outputDir = strcat('Data/Processed/Combined/task',num2str(task),'/',name_folder,'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    % Define the filename to store the data
    outputfilename = sprintf('%s/%s.mat', outputDir, num2str(Number_of_subject));
    % Write it to disk
    save(outputfilename,'Subject');
    clear Subject
end

end

