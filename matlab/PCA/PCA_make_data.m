clear all;


Size_of_subject=105;
task=1;
size_of_vector=225;

for subject_i=1:Size_of_subject

    Submain=load(strcat('Data/PCA/task',num2str(task),'/',num2str(subject_i),'.mat'));
    
    randarray=randi([1 Size_of_subject],size_of_vector-min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)]),1);
    while find( randarray==subject_i)
        randarray=randi([1 Size_of_subject],size_of_vector-size(Submain.Subject_pca.T1,2),1);
        enda
    Subject.cues=zeros(size_of_vector,1);
    for i=1:min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)])
        Subject.cues(i)=1;
    end
    for rand_sub=1:size(randarray,1)
        Random_choose{rand_sub}=load(strcat('Data/PCA/task',num2str(task),'/',num2str(randarray(rand_sub)),'.mat'));
        random_trial(rand_sub)=randi([1 min([size(Random_choose{rand_sub}.Subject_pca.T1,2) size(Random_choose{rand_sub}.Subject_pca.T2,2)])],...
            1,1);
    end
    
    %% T1
    Subject.T1=zeros(size_of_vector,size(Submain.Subject_pca.T1{1},2));
    for Number_trial=1:min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)])
        Subject.T1(Number_trial,:)=Submain.Subject_pca.T1{Number_trial};
    end
    for rand_sub=1:size(randarray,1)
        Number_trial=(min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)])+rand_sub);
        Subject.T1(Number_trial,:)=Random_choose{rand_sub}.Subject_pca.T1{random_trial(rand_sub)};
    end
    
    %% T2
    Subject.T2=zeros(size_of_vector,size(Submain.Subject_pca.T2{1},2));
    for Number_trial=1:min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)])
        Subject.T2(Number_trial,:)=Submain.Subject_pca.T2{Number_trial};
    end
    for rand_sub=1:size(randarray,1)
        Number_trial=(min([size(Submain.Subject_pca.T1,2) size(Submain.Subject_pca.T2,2)])+rand_sub);
        Subject.T2(Number_trial,:)=Random_choose{rand_sub}.Subject_pca.T2{random_trial(rand_sub)};
    end
    Subject.result_label=Submain.Subject_pca.result_label;
    %% mix data
    idx=randperm(numel(Subject.cues));
    Subject.cues(1:end)=Subject.cues(idx);
    for i=1:size(Subject.T2,2)
        Subject.T2(1:end,i)=Subject.T2(idx,i);
        Subject.T1(1:end,i)=Subject.T1(idx,i);
    end
    
    %% save
    outputDir = strcat('Data/PCA_SVM/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,num2str(subject_i),'.mat'),'Subject');
    
end