%% set
clear all;
task=1;%set number of task
size_of_subjects=105;%set


for Number_of_subject=1:size_of_subjects

    name_file=strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    load(char(name_file));
    for i=1:18
        for j=1:size(Subject.result_label,2)
            Subject_re.result_label{i,j}=Subject.result_label{i+21,j};
        end
    end
    for i=19:39
        for j=1:size(Subject.result_label,2)
            Subject_re.result_label{i,j}=Subject.result_label{i-18,j};
        end
    end
    for i=40:64
        for j=1:size(Subject.result_label,2)
            Subject_re.result_label{i,j}=Subject.result_label{i,j};
        end
    end
    
    % T1
    for Number_of_trial=1:size( Subject.T1,2)
        for i=1:18
            for j=1:size(Subject.T1{Number_of_trial},2)
                Subject_re.T1{Number_of_trial}(i,j)=Subject.T1{Number_of_trial}(i+21,j);
            end
        end
        for i=19:39
            for j=1:size(Subject.T1{Number_of_trial},2)
                Subject_re.T1{Number_of_trial}(i,j)=Subject.T1{Number_of_trial}(i-18,j);
            end
        end
        for i=40:64
            for j=1:size(Subject.T1{Number_of_trial},2)
                Subject_re.T1{Number_of_trial}(i,j)=Subject.T1{Number_of_trial}(i,j);
            end
        end
    end
    %T2
    for Number_of_trial=1:size( Subject.T2,2)
        for i=1:18
            for j=1:size(Subject.T2{Number_of_trial},2)
                Subject_re.T2{Number_of_trial}(i,j)=Subject.T2{Number_of_trial}(i+21,j);
            end
        end
        for i=19:39
            for j=1:size(Subject.T2{Number_of_trial},2)
                Subject_re.T2{Number_of_trial}(i,j)=Subject.T2{Number_of_trial}(i-18,j);
            end
        end
        for i=40:64
            for j=1:size(Subject.T2{Number_of_trial},2)
                Subject_re.T2{Number_of_trial}(i,j)=Subject.T2{Number_of_trial}(i,j);
            end
        end
    end
    Subject_re.Subject_old=Subject;
    clear Suject;
    Subject.T1=Subject_re.T1;
    Subject.T2=Subject_re.T2;
    Subject.result_label=Subject_re.result_label;
    Subject.Subject_old=Subject_re.Subject_old;
    %% Saving
    jsonStr = jsonencode(Subject); 
    fid = fopen(strcat('Result_json/Task',num2str(task),'/',num2str(Number_of_subject),'.json'), 'w'); 
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



