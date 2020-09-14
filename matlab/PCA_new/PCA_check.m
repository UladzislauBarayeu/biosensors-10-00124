function [] = PCA_check( task, fast, fast_check, knn, selected_channels,...
    number_sub_channel, threshold, KernelSVM, size_of_subject)
%PCA_CHECK check models
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================

if strcmp(selected_channels{1},'64_channels')
    name_folder='64_channels';
else
    name_folder=strcat(num2str(length(selected_channels)),'_channels');
end

predict_both_posterior_all=[];
predict_T1_posterior_all=[];
predict_T2_posterior_all=[];
labels_all=[];


%%
for subject_i=1:size_of_subject
    %make train model
    data=load(strcat('Data/PCA_SVM/task',num2str(task),'/',name_folder,'/',num2str(subject_i),'.mat'));
    XT1=data.Subject.T1;
    XT2=data.Subject.T2;
    Y=data.Subject.cues;
    
    nbFolds = 5;
    
    uniqueTrials = (1:size(XT1,1));
    nbTrials = numel(uniqueTrials);            
    assert(mod(nbTrials,nbFolds) == 0);
    foldSize = nbTrials / nbFolds;
    folds = (0 : foldSize : nbTrials);

    errorIT1=zeros(1,nbFolds);
    errorIIT1=zeros(1,nbFolds);
    errorIT2=zeros(1,nbFolds);
    errorIIT2=zeros(1,nbFolds);
    errorIboth=zeros(1,nbFolds);
    errorIIboth=zeros(1,nbFolds);

    for f = 1 : nbFolds
        
        if fast
            if knn
                resT1=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/fast/T1/knn',num2str(knn),'/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            else
                resT1=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/fast/T1/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            end
        else
            if knn
                resT1=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/slow/T1/knn',num2str(knn),'/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            else
                resT1=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/slow/T1/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            end
        end
        [~,ind_max_T1]=max(resT1.result_accuracy);
        if fast
            if knn
                resT2=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/fast/T2/knn',num2str(knn),'/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            else
                resT2=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/fast/T2/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            end
        else
            if knn
                resT2=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/slow/T2/knn',num2str(knn),'/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            else
                resT2=load(strcat('Data/PCA_results/task',num2str(task),'/',...
                    name_folder,'/slow/T2/',KernelSVM,'/',num2str(subject_i),'_',num2str(f),'.mat'));
            end
        end
        [~,ind_max_T2]=max(resT2.result_accuracy);
    
        %%
        errorIIT1_test=zeros(1,size_of_subject-1);
        errorIIT2_test=zeros(1,size_of_subject-1);
        errorIIboth_test=zeros(1,size_of_subject-1);
        
        %% T1
        FIdx_T1=resT1.Indexes{ind_max_T1}{1}; 
        testMaskT1 = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
        testCuesT1 = Y(testMaskT1);
        testTrialsT1 = XT1(testMaskT1, :, :);
        trainCuesT1 = Y(~testMaskT1);
        trainTrialsT1 = XT1(~testMaskT1, :, :);
        
        
        trainTrials_pca=zeros(size(trainTrialsT1,1),size(trainTrialsT1,2)*number_sub_channel);
        testTrials_pca=zeros(size(testTrialsT1,1),size(testTrialsT1,2)*number_sub_channel);
        for pca_i=1:size(trainTrialsT1,2)
            pca_data_train=zeros(size(trainTrialsT1,1),size(trainTrialsT1,3));
            pca_data_test=zeros(size(testTrialsT1,1),size(testTrialsT1,3));

            for pca_j=1:size(trainTrialsT1,1)
                for pca_k=1:size(trainTrialsT1,3)
                    pca_data_train(pca_j,pca_k)=trainTrialsT1(pca_j,pca_i,pca_k);
                end
            end
            [COEFFT1{pca_i}, SCORE, ~, ~, ~, MUT1{pca_i}]=pca(pca_data_train);

            score_test=zeros(size(testTrialsT1,1),size(COEFFT1{pca_i},2));
            for pca_j=1:size(testTrialsT1,1)
                for pca_k=1:size(testTrialsT1,3)
                    pca_data_test(pca_j,pca_k)=testTrialsT1(pca_j,pca_i,pca_k);
                end
                pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MUT1{pca_i};
                score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFFT1{pca_i}';
            end
            for sub_channel_i=1:number_sub_channel
                trainTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=SCORE(:,sub_channel_i);
                testTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
            end
        end
        trainTrialsT1=trainTrials_pca;
        testTrialsT1=testTrials_pca;
                

        trainTrialsT1 = trainTrialsT1(:,FIdx_T1);
        testTrialsT1 = testTrialsT1(:,FIdx_T1);

        %normalize train data
        %get max and min for train data
        trainTrialsT1=trainTrialsT1.';
        testTrialsT1=testTrialsT1.';
        for normit=1:size(trainTrialsT1,1)
            [trainTrialsT1(normit,:),max_train_T1(normit), min_train_T1(normit)]=...
                normalize_me(trainTrialsT1(normit, :));

            for test_iterator=1:size(testTrialsT1,2)
                testTrialsT1(normit, test_iterator)=(testTrialsT1(normit,test_iterator)-min_train_T1(normit))...
                    /(max_train_T1(normit)-min_train_T1(normit));
            end
        end
        trainTrialsT1=trainTrialsT1.';
        testTrialsT1=testTrialsT1.';
        


        % Classifier
        if knn
            ModelT1 = fitcknn(trainTrialsT1,trainCuesT1,'NumNeighbors',knn,'Standardize',1);% num_of neubors?
        else
            if fast_check
                ModelT1 = fitcsvm(trainTrialsT1,trainCuesT1,'Standardize',true,...
                        'KernelFunction',KernelSVM,'KernelScale','auto');
            else
                opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 10);
                ModelT1 = fitcsvm(trainTrialsT1, trainCuesT1,'Standardize',true,...
                    'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
            end
        end
        
        %posterior
        if knn
            [predictT1,predictT1_posterior,~] = predict(ModelT1,testTrialsT1);
            %predictT1 = ModelT1.predict(testTrialsT1);
        else
            predictT1 = ModelT1.predict(testTrialsT1);
            ModelT1_posterior = fitPosterior(ModelT1);
            [~,predictT1_posterior] = predict(ModelT1_posterior,testTrialsT1);
        end
        for i=1:length(predictT1)
            if predictT1_posterior(i,2)>threshold
                predictT1(i)=1;
            else
                predictT1(i)=0;
            end
        end
        
        %% T2
        FIdx_T2=resT2.Indexes{ind_max_T2}{1}; 
        testMaskT2 = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
        testCuesT2 = Y(testMaskT2);
        testTrialsT2 = XT2(testMaskT2, :, :);
        trainCuesT2 = Y(~testMaskT2);
        trainTrialsT2 = XT2(~testMaskT2, :, :);
        
                
        
        trainTrials_pca=zeros(size(trainTrialsT2,1),size(trainTrialsT2,2)*number_sub_channel);
        testTrials_pca=zeros(size(testTrialsT2,1),size(testTrialsT2,2)*number_sub_channel);
        for pca_i=1:size(trainTrialsT2,2)
            pca_data_train=zeros(size(trainTrialsT2,1),size(trainTrialsT2,3));
            pca_data_test=zeros(size(testTrialsT2,1),size(testTrialsT2,3));

            for pca_j=1:size(trainTrialsT2,1)
                for pca_k=1:size(trainTrialsT2,3)
                    pca_data_train(pca_j,pca_k)=trainTrialsT2(pca_j,pca_i,pca_k);
                end
            end
            [COEFFT2{pca_i}, SCORE, ~, ~, ~, MUT2{pca_i}]=pca(pca_data_train);

            score_test=zeros(size(testTrialsT2,1),size(COEFFT2{pca_i},2));
            for pca_j=1:size(testTrialsT2,1)
                for pca_k=1:size(testTrialsT2,3)
                    pca_data_test(pca_j,pca_k)=testTrialsT2(pca_j,pca_i,pca_k);
                end
                pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MUT2{pca_i};
                score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFFT2{pca_i}';
            end
            for sub_channel_i=1:number_sub_channel
                trainTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=SCORE(:,sub_channel_i);
                testTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
            end
        end
        trainTrialsT2=trainTrials_pca;
        testTrialsT2=testTrials_pca;

        trainTrialsT2 = trainTrialsT2(:,FIdx_T2);
        testTrialsT2 = testTrialsT2(:,FIdx_T2);

        %normalize train data
        %get max and min for train data
        trainTrialsT2=trainTrialsT2.';
        testTrialsT2=testTrialsT2.';
        for normit=1:size(trainTrialsT2,1)
            [trainTrialsT2(normit,:),max_train_T2(normit), min_train_T2(normit)]=...
                normalize_me(trainTrialsT2(normit, :));

            for test_iterator=1:size(testTrialsT2,2)
                testTrialsT2(normit, test_iterator)=(testTrialsT2(normit,test_iterator)-min_train_T2(normit))...
                    /(max_train_T2(normit)-min_train_T2(normit));
            end
        end
        trainTrialsT2=trainTrialsT2.';
        testTrialsT2=testTrialsT2.';

        % Classifier
        if knn
            ModelT2 = fitcknn(trainTrialsT2,trainCuesT2,'NumNeighbors',knn,'Standardize',1);% num_of neubors?
        else
            if fast_check
                ModelT2 = fitcsvm(trainTrialsT2,trainCuesT2,'Standardize',true,...
                        'KernelFunction',KernelSVM,'KernelScale','auto');
            else
                opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 10);
                ModelT2 = fitcsvm(trainTrialsT2, trainCuesT2,'Standardize',true,...
                    'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
            end
        end
        if knn
            [predictT2,predictT2_posterior,~] = predict(ModelT2,testTrialsT2);
        else
            predictT2 = ModelT2.predict(testTrialsT2);
            %posterior
            ModelT2_posterior = fitPosterior(ModelT2);
            [~,predictT2_posterior] = predict(ModelT2_posterior,testTrialsT2);
        end  
        for i=1:length(predictT2)
            if predictT2_posterior(i,2)>threshold
                predictT2(i)=1;
            else
                predictT2(i)=0;
            end
        end
        

        
        
        for i=1:size(testCuesT1,1)

            if predictT1(i)==1 && predictT2(i)==1
                predict_both(i)=1;
            else
                predict_both(i)=0;
            end
            
            [~,more_sceptic]=min([predictT1_posterior(i,2) predictT2_posterior(i,2)]);
            if more_sceptic==1
                predict_both_posterior(i)=predictT1_posterior(i,2);
            else
                predict_both_posterior(i)=predictT2_posterior(i,2);
            end
            

            %T1
            good=sum(testCuesT1);
            if testCuesT1(i)==1 && predictT1(i)==0
                errorIT1(f)=errorIT1(f)+1/good;
            end
            if testCuesT1(i)==0 && predictT1(i)==1
                errorIIT1(f)=errorIIT1(f)+1/(size(testCuesT1,1)-good);
            end

            %T2
            if testCuesT2(i)==1 && predictT2(i)==0
                errorIT2(f)=errorIT2(f)+1/good;
            end
            if testCuesT2(i)==0 && predictT2(i)==1
                errorIIT2(f)=errorIIT2(f)+1/(size(testCuesT1,1)-good);
            end
            
            %both
            if testCuesT1(i)==1 && predict_both(i)==0
                errorIboth(f)=errorIboth(f)+1/good;
            end
            if testCuesT1(i)==0 && predict_both(i)==1
                errorIIboth(f)=errorIIboth(f)+1/(size(testCuesT1,1)-good);
            end
        end
        labels={};
        for i_label=1:length(testCuesT1)
            if testCuesT1(i_label)
                labels{i_label}='subject';
            else
                labels{i_label}='others.';
            end
            
        end
        ACCT1(f) = mean(predictT1 == testCuesT1);
        ACCT2(f) = mean(predictT2 == testCuesT2);
        ACC(f) = mean(predict_both.' == testCuesT1);
        iterator=1;
        
        prob_test_both=[];
        for subject_i_test=1:size_of_subject
            if subject_i_test==subject_i
                continue
            else
                Subtest=load(strcat('Data/PCA_SVM/task',num2str(task),'/',name_folder,'/',num2str(subject_i_test),'.mat'));
                T1_cues=zeros(size(Subtest.Subject.T1,1)/2,1);
                %% change bug
                indexis=[];
                for ind_i=1:size(Subtest.Subject.T1,1)
                    if Subtest.Subject.cues(ind_i)==1
                        indexis=[indexis ind_i];
                    end
                end
                % T1
                testTrialsT1=Subtest.Subject.T1(indexis,:,:);
                T1_test_pca=zeros(size(testTrialsT1,1),size(testTrialsT1,2)*number_sub_channel);
                for pca_i=1:size(testTrialsT1,2)
                    pca_data_test=zeros(size(testTrialsT1,1),size(testTrialsT1,3));

                    score_test=zeros(size(testTrialsT1,1),size(COEFFT1{pca_i},2));
                    for pca_j=1:size(testTrialsT1,1)
                        for pca_k=1:size(testTrialsT1,3)
                            pca_data_test(pca_j,pca_k)=testTrialsT1(pca_j,pca_i,pca_k);
                        end
                        pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MUT1{pca_i};
                        score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFFT1{pca_i}';
                    end
                    for sub_channel_i=1:number_sub_channel
                        T1_test_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
                    end
                end
                T1_test=T1_test_pca;

                T1_test = T1_test(:,FIdx_T1);
                for test_trial=1:size(T1_test,1)
                    for test_iterator=1:size(T1_test,2)
                        T1_test(test_trial, test_iterator)=(T1_test(test_trial,test_iterator)-min_train_T1(test_iterator))...
                            /(max_train_T1(test_iterator)-min_train_T1(test_iterator));
                    end
                end
                % T2
                testTrialsT2=Subtest.Subject.T2(indexis,:,:);
                T2_test_pca=zeros(size(testTrialsT2,1),size(testTrialsT2,2)*number_sub_channel);
                for pca_i=1:size(testTrialsT2,2)
                    pca_data_test=zeros(size(testTrialsT2,1),size(testTrialsT2,3));

                    score_test=zeros(size(testTrialsT2,1),size(COEFFT2{pca_i},2));
                    for pca_j=1:size(testTrialsT2,1)
                        for pca_k=1:size(testTrialsT2,3)
                            pca_data_test(pca_j,pca_k)=testTrialsT2(pca_j,pca_i,pca_k);
                        end
                        pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MUT2{pca_i};
                        score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFFT2{pca_i}';
                    end
                    for sub_channel_i=1:number_sub_channel
                        T2_test_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
                    end
                end
                T2_test=T2_test_pca;
                
                T2_test = T2_test(:,FIdx_T2);
                for test_trial=1:size(T2_test,1)
                    for test_iterator=1:size(T2_test,2)
                        T2_test(test_trial, test_iterator)=(T2_test(test_trial,test_iterator)-min_train_T2(test_iterator))...
                            /(max_train_T2(test_iterator)-min_train_T2(test_iterator));
                    end
                end
                
                
                %SVM
                predictT1_test = ModelT1.predict(T1_test);
                predictT2_test = ModelT2.predict(T2_test);
                %posterior
                [~,predictT1_test_posterior] = predict(ModelT1_posterior,T1_test);
                [~,predictT2_test_posterior] = predict(ModelT2_posterior,T2_test);
                for i=1:length(predictT1_test)
                    if predictT1_test_posterior(i,2)>threshold
                        predictT1_test(i)=1;
                    else
                        predictT1_test(i)=0;
                    end
                end
                for i=1:length(predictT2_test)
                    if predictT2_test_posterior(i,2)>threshold
                        predictT2_test(i)=1;
                    else
                        predictT2_test(i)=0;
                    end
                end
                for i=1:size(predictT1_test,1)
                    if predictT1_test(i)==1 && predictT2_test(i)==1
                        predict_test(i)=1;
                    else
                        predict_test(i)=0;
                    end
                    %T1
                    if predictT1_test(i)==1
                        errorIIT1_test(iterator)=errorIIT1_test(iterator)+1/size(T1_cues,1);
                    end

                    %T2
                    if predictT2_test(i)==1
                        errorIIT2_test(iterator)=errorIIT2_test(iterator)+1/size(T1_cues,1);
                    end

                    %both
                    if predict_test(i)==1
                        errorIIboth_test(iterator)=errorIIboth_test(iterator)+1/size(T1_cues,1);
                    end
                    
                    
                end
                iterator=iterator+1;
                for i=1:size(predictT1_test_posterior,1)
                    [~,more_sceptic]=min([predictT1_test_posterior(i,2) predictT2_test_posterior(i,2)]);
                    if more_sceptic==1
                        predict_both_test_posterior(i)=predictT1_test_posterior(i,2);
                    else
                        predict_both_test_posterior(i)=predictT2_test_posterior(i,2);
                    end
                end
                prob_test_both=[prob_test_both predict_both_test_posterior];
            end
        end
        errorIIT1_test_fold(f)=mean(errorIIT1_test);
        errorIIT2_test_fold(f)=mean(errorIIT2_test);
        errorIIboth_test_fold(f)=mean(errorIIboth_test);
        
        %ROC
        for i=1:length(predict_both_posterior)
            if strcmp(labels{i},'others.')
                predict_both_posterior(i)=prob_test_both(randi([1 length(prob_test_both)]));
            end
        end
        predict_both_posterior_all=[predict_both_posterior_all predict_both_posterior];
        
        predict_T1_posterior_all=[predict_T1_posterior_all predictT1_posterior(:,2).'];
        predict_T2_posterior_all=[predict_T2_posterior_all predictT2_posterior(:,2).'];

        labels_all=[labels_all labels];
        
    end

    accuracyT1(subject_i)=mean(ACCT1);
    accuracyT2(subject_i)=mean(ACCT2);
    accuracyBoth(subject_i)=mean(ACC);

    ErrorI_T1(subject_i)=mean(errorIT1);
    ErrorII_T1(subject_i)=mean(errorIIT1);

    ErrorI_T2(subject_i)=mean(errorIT2);
    ErrorII_T2(subject_i)=mean(errorIIT2);

    ErrorI_both(subject_i)=mean(errorIboth);
    ErrorII_both(subject_i)=mean(errorIIboth);
    
    
    ErrorIIT1_test(subject_i)=mean(errorIIT1_test_fold);
    ErrorIIT2_test(subject_i)=mean(errorIIT2_test_fold);
    ErrorIITboth_test(subject_i)=mean(errorIIboth_test_fold);
end

[X_all,Y_all,~,AUC_all] = perfcurve(labels_all,predict_both_posterior_all,'subject');
plot(X_all,Y_all)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for two action')
%save file
outputjpgDir = strcat('figures/ROC/PCA/task',num2str(task),'/',name_folder,'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-both.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);

[X_T1,Y_T1,~,~] = perfcurve(labels_all,predict_T1_posterior_all,'subject');
plot(X_T1,Y_T1)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for T1')
namefile=strcat('%s','-ROC-T1.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);


[X_T2,Y_T2,~,~] = perfcurve(labels_all,predict_T2_posterior_all,'subject');
plot(X_T2,Y_T2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for T2')

namefile=strcat('%s','-ROC-T2.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);


AccT1=mean(accuracyT1);AccT2=mean(accuracyT2);AccBoth=mean(accuracyBoth);
ErrI_T1=mean(ErrorI_T1);ErrII_T1=mean(ErrorII_T1);
ErrI_T2=mean(ErrorI_T2);ErrII_T2=mean(ErrorII_T2);
ErrI_both=mean(ErrorI_both);ErrII_both=mean(ErrorII_both);
%test
ErrII_T1_test=mean(ErrorIIT1_test);ErrII_T2_test=mean(ErrorIIT2_test);ErrII_both_test=mean(ErrorIITboth_test);
%% save
if knn
    if fast
        outputDir = strcat('Data/PCA_final/task',num2str(task),'/',name_folder,'/fast/knn',num2str(knn),'/');
    else
        outputDir = strcat('Data/PCA_final/task',num2str(task),'/',name_folder,'/slow/knn',num2str(knn),'/');
    end
else
    if fast
        outputDir = strcat('Data/PCA_final/task',num2str(task),'/',name_folder,'/fast/');
    else
        outputDir = strcat('Data/PCA_final/task',num2str(task),'/',name_folder,'/slow/');
    end
end
% Check if the folder exists , and if not, make it...
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'result_PCA.mat'),'AccT1','AccT2','AccBoth','accuracyT1','accuracyT2','accuracyBoth'...
    ,'ErrI_T1','ErrII_T1','ErrorI_T1','ErrorII_T1'...
    ,'ErrI_T2','ErrII_T2','ErrorI_T2','ErrorII_T2'...
    ,'ErrI_both','ErrII_both','ErrorI_both','ErrorII_both'...
    ,'ErrII_T1_test','ErrII_T2_test','ErrII_both_test'...
    ,'ErrorIIT1_test','ErrorIIT2_test','ErrorIITboth_test'...
    ,'X_all','Y_all','X_T1','Y_T1','X_T2','Y_T2'...
    ,'AUC_all');


end

