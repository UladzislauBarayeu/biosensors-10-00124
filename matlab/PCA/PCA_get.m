function [] = PCA_get( task, number_sub_channel)
%PCA_GET convolute electrodes for each feature

size_of_subjects=105;

for Number_of_subject=1:size_of_subjects
    name_file=strcat('Data\Processed\Combined\task',num2str(task),'/',num2str(Number_of_subject),'.mat');
    load(char(name_file));
    
    %% T1
    data_PCA={};
    for numer_feat=1:size(Subject.T1{1},2)
        Input_for_PCA=zeros(size(Subject.T1,2),size(Subject.T1{1},1));
        for number_trial=1:size(Subject.T1,2)
            for number_channel=1:size(Subject.T1{1},1)
                Input_for_PCA(number_trial,number_channel)=Subject.T1{number_trial}(number_channel,numer_feat);
            end
        end
        coeff=[];score=[];
        [coeff,score]=pca(Input_for_PCA);
        data_PCA{numer_feat} = score(:,1:number_sub_channel);
    end

    for number_trial=1:size(Subject.T1,2)
        for number_channel=1:size(data_PCA{1},2)
            for numer_feat=1:size(Subject.T1{1},2)
                Subject_pca.T1{number_trial}((number_channel)+(numer_feat-1)*2)=data_PCA{numer_feat}(number_trial,number_channel);
            end
        end
    end
    
    %% T2
    data_PCA={};
    for numer_feat=1:size(Subject.T2{1},2)
        Input_for_PCA=zeros(size(Subject.T2,2),size(Subject.T2{1},1));
        for number_trial=1:size(Subject.T2,2)
            for number_channel=1:size(Subject.T2{1},1)
                Input_for_PCA(number_trial,number_channel)=Subject.T2{number_trial}(number_channel,numer_feat);
            end
        end
        coeff=[];score=[];
        [coeff,score]=pca(Input_for_PCA);
        data_PCA{numer_feat} = score(:,1:number_sub_channel);
    end

    for number_trial=1:size(Subject.T2,2)
        for number_channel=1:size(data_PCA{1},2)
            for numer_feat=1:size(Subject.T2{1},2)
                Subject_pca.T2{number_trial}((number_channel)+(numer_feat-1)*2)=data_PCA{numer_feat}(number_trial,number_channel);
            end
        end
    end
    %% label
    for number_channel=1:size(data_PCA{1},2)
        for numer_feat=1:size(Subject.T2{1},2)/2
            Subject_pca.result_label{(number_channel)+(numer_feat-1)*2}=Subject.result_label{1,numer_feat}(6:end);
        end
    end
    for number_channel=1:size(data_PCA{1},2)
        for numer_feat=(size(Subject.T2{1},2)/2+1):size(Subject.T2{1},2)
            name=char(Subject.result_label{1,numer_feat});
            Subject_pca.result_label{(number_channel)+(numer_feat-1)*2}=name(6:end);
        end
    end
    %% save
    outputDir = strcat('Data/PCA/task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,num2str(Number_of_subject),'.mat'),'Subject_pca');
end


end






