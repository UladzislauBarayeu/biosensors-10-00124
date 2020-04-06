function [] = PCA_make_data_new( task, size_of_vector, selected_channels )


Size_of_subject=105;
name_folder='';
for nb_channel=1:size( selected_channels,2)
    name_folder=strcat(name_folder,selected_channels{nb_channel});
end
name_folder=strcat(name_folder,'end');

for subject_i=1:Size_of_subject
    name_file=strcat('Data/Processed/Combined/task',num2str(task),'/',name_folder,'/',num2str(subject_i),'.mat');
    Submain=load(char(name_file));
    
    randarray=randi([1 Size_of_subject],size_of_vector-min(size(Submain.Subject.T1,2), size(Submain.Subject.T2,2)),1);
    while find( randarray==subject_i)
        randarray=randi([1 Size_of_subject],size_of_vector-min(size(Submain.Subject.T1,2), size(Submain.Subject.T2,2)),1);
    end
    Subject.cues=[ones(1,min(size(Submain.Subject.T1,2), size(Submain.Subject.T2,2)))...
        zeros(1,size_of_vector-min(size(Submain.Subject.T1,2), size(Submain.Subject.T2,2)))];
    Subject.cues=Subject.cues.';
    for rand_sub=1:size(randarray,1)
        Random_choose{rand_sub}=load(strcat('Data/Processed/Combined/task',num2str(task),'/',name_folder,'/',...
            num2str(randarray(rand_sub)),'.mat'));
        minval=min(size(Random_choose{rand_sub}.Subject.T1,2), size(Random_choose{rand_sub}.Subject.T2,2));
        random_trial(rand_sub)=randi([1 minval],1,1);
    end
    
    Subject.T1=zeros(size_of_vector,size(Submain.Subject.T1{1},2),size(Submain.Subject.T1{1},1));
    Subject.T2=zeros(size_of_vector,size(Submain.Subject.T2{1},2),size(Submain.Subject.T2{1},1));
    for trial=1:(size_of_vector-size(randarray,1))
        Subject.T1(trial,:,:)=Submain.Subject.T1{trial}.';
        Subject.T2(trial,:,:)=Submain.Subject.T2{trial}.';
    end
    
    for trial=1:size(randarray,1)
        Subject.T1((size_of_vector-size(randarray,1))+trial,:,:)=...
            Random_choose{trial}.Subject.T1{random_trial(trial)}.';
        Subject.T2((size_of_vector-size(randarray,1))+trial,:,:)=... 
            Random_choose{trial}.Subject.T2{random_trial(trial)}.';
    end
    
    %% mix data
    idx=randperm(numel(Subject.cues));
    Subject.cues(1:end)=Subject.cues(idx);
    for i=1:size(Subject.T2,2)
        for j=1:size(Subject.T2,3)
            Subject.T2(1:end,i,j)=Subject.T2(idx,i,j);
            Subject.T1(1:end,i,j)=Subject.T1(idx,i,j);
        end
    end
    
    %% save
    outputDir = strcat('Data/PCA_SVM/task',num2str(task),'/',name_folder,'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,num2str(subject_i),'.mat'),'Subject');
    clearvars -except subject_i task size_of_vector selected_channels name_folder Size_of_subject
end


end






