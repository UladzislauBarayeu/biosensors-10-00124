function [] = Extract_EMD_feature( task, type, number_of_IMFS )
%EXTRACT_EMD_FEATURE 
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
size_of_subjects=105;

%%
T1_EMD_par={};
T2_EMD_par={};
labels_par={};
for Number_of_subject=1:size_of_subjects
    
    name_file=strcat('Data/preproces/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    
    preProcesed=load(char(name_file));
    [T1_EMD_par{Number_of_subject},labels_par{Number_of_subject}]=...
        EMD(preProcesed.T1, preProcesed.channel_names, type, number_of_IMFS);
    [T2_EMD_par{Number_of_subject},~]=...
        EMD(preProcesed.T2, preProcesed.channel_names, type, number_of_IMFS);
    sample_rate_new_par{Number_of_subject}=preProcesed.sample_rate;
    
    T1_EMD=T1_EMD_par{Number_of_subject};
    T2_EMD=T2_EMD_par{Number_of_subject};
    labels=labels_par{Number_of_subject};
    sample_rate_new=sample_rate_new_par{Number_of_subject};
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

end

