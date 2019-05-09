clear all;






%% set
size_of_subject=105;
task=1;
fast=1;
fast_check=1;

predict_both_posterior_all=[];
predict_T1_posterior_all=[];
predict_T2_posterior_all=[];
labels_all=[];


%%
for subject_i=1:size_of_subject
    if fast
        resT1=load(strcat('Data/PCA_results/task',num2str(task),'/fast/T1/',num2str(subject_i),'.mat'));
    else
        resT1=load(strcat('Data/PCA_results/task',num2str(task),'/slow/T1/',num2str(subject_i),'.mat'));
    end
    [~,ind_max_T1]=max(resT1.result_accuracy);
    if fast
        resT2=load(strcat('Data/PCA_results/task',num2str(task),'/fast/T2/',num2str(subject_i),'.mat'));
    else
        resT2=load(strcat('Data/PCA_results/task',num2str(task),'/slow/T2/',num2str(subject_i),'.mat'));
    end
    [~,ind_max_T2]=max(resT2.result_accuracy);

    %make train model
    data=load(strcat('Data/PCA_SVM/task',num2str(task),'/',num2str(subject_i),'.mat'));
    XT1=data.Subject.T1;
    XT2=data.Subject.T2;
    Y=data.Subject.cues;
    
    nbFolds = 5;
    KernelSVM='rbf';%'rbf' or 'linear' optional
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
        errorIIT1_test=zeros(1,size_of_subject-1);
        errorIIT2_test=zeros(1,size_of_subject-1);
        errorIIboth_test=zeros(1,size_of_subject-1);
        %% T1
        FIdx_T1=resT1.Indexes{ind_max_T1}{1}; 
        testMaskT1 = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
        testCuesT1 = Y(testMaskT1);
        testTrialsT1 = XT1(testMaskT1, :);
        trainCuesT1 = Y(~testMaskT1);
        trainTrialsT1 = XT1(~testMaskT1, :);

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

        % SVM Classifier
        if fast_check
            SVMModelT1 = fitcsvm(trainTrialsT1,trainCuesT1,'Standardize',true,...
                    'KernelFunction',KernelSVM,'KernelScale','auto');
        else
            opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 10);
            SVMModelT1 = fitcsvm(trainTrialsT1, trainCuesT1,'Standardize',true,...
                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
        end
        predictT1 = SVMModelT1.predict(testTrialsT1);
        %posterior
        SVMModelT1_posterior = fitPosterior(SVMModelT1);
        [~,predictT1_posterior] = predict(SVMModelT1_posterior,testTrialsT1);
        
        %% T2
        FIdx_T2=resT2.Indexes{ind_max_T2}{1}; 
        testMaskT2 = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
        testCuesT2 = Y(testMaskT2);
        testTrialsT2 = XT2(testMaskT2, :);
        trainCuesT2 = Y(~testMaskT2);
        trainTrialsT2 = XT2(~testMaskT2, :);

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

        % SVM Classifier
        if fast_check
            SVMModelT2 = fitcsvm(trainTrialsT2,trainCuesT2,'Standardize',true,...
                    'KernelFunction',KernelSVM,'KernelScale','auto');
        else
            opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 10);
            SVMModelT2 = fitcsvm(trainTrialsT2, trainCuesT2,'Standardize',true,...
                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
        end
        predictT2 = SVMModelT2.predict(testTrialsT2);
        %posterior
        SVMModelT2_posterior = fitPosterior(SVMModelT2);
        [~,predictT2_posterior] = predict(SVMModelT2_posterior,testTrialsT2);
                

        for i=1:size(testCuesT1,1)

            if predictT1(i)==1 && predictT2(i)==1
                predict_both(i)=1;
            else
                predict_both(i)=0;
            end
            
            %posterior both
            [~,more_sceptic]=min([predictT1_posterior(i,2) predictT2_posterior(i,2)]);
            if more_sceptic==1
                predict_both_posterior(i)=predictT1_posterior(i,2);
            else
                predict_both_posterior(i)=predictT2_posterior(i,2);
            end

            %T1
            if testCuesT1(i)==1 && predictT1(i)==0
                errorIT1(f)=errorIT1(f)+1/size(testCuesT1,1);
            end
            if testCuesT1(i)==0 && predictT1(i)==1
                errorIIT1(f)=errorIIT1(f)+1/size(testCuesT1,1);
            end

            %T2
            if testCuesT2(i)==1 && predictT2(i)==0
                errorIT2(f)=errorIT2(f)+1/size(testCuesT1,1);
            end
            if testCuesT2(i)==0 && predictT2(i)==1
                errorIIT2(f)=errorIIT2(f)+1/size(testCuesT1,1);
            end
            
            %both
            if testCuesT1(i)==1 && predict_both(i)==0
                errorIboth(f)=errorIboth(f)+1/size(testCuesT1,1);
            end
            if testCuesT1(i)==0 && predict_both(i)==1
                errorIIboth(f)=errorIIboth(f)+1/size(testCuesT1,1);
            end
        end
        for i_label=1:length(testCuesT1)
            if testCuesT1(i_label)
                labels{i_label}='subject'; 
            else
                labels{i_label}='other'; 
            end
            
        end
        %predict_both_posterior=predict_both_posterior.';
        %labels=labels.';
        %[X,Y,T,AUC] = perfcurve(labels,predict_both_posterior,'subject');
        %plot(X,Y)
        %xlabel('False positive rate') 
        %ylabel('True positive rate')
        %title('ROC for Classification by Logistic Regression')
        ACCT1(f) = mean(predictT1 == testCuesT1);
        ACCT2(f) = mean(predictT2 == testCuesT1);
        ACC(f) = mean(predict_both.' == testCuesT1);
        iterator=1;
        for subject_i_test=1:size_of_subject
            if subject_i_test==subject_i
                continue
            else
                Subtest=load(strcat('Data/PCA/task',num2str(task),'/',num2str(subject_i_test),'.mat'));
                T1_test=zeros(min(size(Subtest.Subject_pca.T1,2),size(Subtest.Subject_pca.T2,2))...
                    ,size(Subtest.Subject_pca.T1{1},2));
                T2_test=zeros(min(size(Subtest.Subject_pca.T1,2),size(Subtest.Subject_pca.T2,2))...
                    ,size(Subtest.Subject_pca.T2{1},2));
                T1_cues=zeros(size(T1_test,1),1);
                % T1
                for test_trial=1:size(T1_test,1)
                    T1_test(test_trial,:)=Subtest.Subject_pca.T1{test_trial};
                    
                end
                T1_test = T1_test(:,FIdx_T1);
                for test_trial=1:size(T1_test,1)
                    for test_iterator=1:size(T1_test,2)
                        T1_test(test_trial, test_iterator)=(T1_test(test_trial,test_iterator)-min_train_T1(test_iterator))...
                            /(max_train_T1(test_iterator)-min_train_T1(test_iterator));
                    end
                end
                % T2
                for test_trial=1:size(T2_test,1)
                    T2_test(test_trial,:)=Subtest.Subject_pca.T2{test_trial};
                end
                T2_test = T2_test(:,FIdx_T2);
                for test_trial=1:size(T2_test,1)
                    for test_iterator=1:size(T2_test,2)
                        T2_test(test_trial, test_iterator)=(T2_test(test_trial,test_iterator)-min_train_T2(test_iterator))...
                            /(max_train_T2(test_iterator)-min_train_T2(test_iterator));
                    end
                end
                
                
                %SVM
                predictT1_test = SVMModelT1.predict(T1_test);
                predictT2_test = SVMModelT2.predict(T2_test);
                %posterior
                [~,predictT1_test_posterior] = predict(SVMModelT1_posterior,T1_test);
                [~,predictT2_test_posterior] = predict(SVMModelT2_posterior,T2_test);
                
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
            end
        end
        errorIIT1_test_fold(f)=mean(errorIIT1_test);
        errorIIT2_test_fold(f)=mean(errorIIT2_test);
        errorIIboth_test_fold(f)=mean(errorIIboth_test);
        
        %ROC
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

[X_all,Y_all,~,~] = perfcurve(labels_all,predict_both_posterior_all,'subject');
plot(X_all,Y_all)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for two action')
%save file
outputjpgDir = strcat('figures/ROC/PCA/');
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
%save file
outputjpgDir = strcat('figures/ROC/PCA/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-T1.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);


[X_T2,Y_T2,~,~] = perfcurve(labels_all,predict_T2_posterior_all,'subject');
plot(X_T2,Y_T2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for T2')
%save file
outputjpgDir = strcat('figures/ROC/PCA/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
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
if fast
    outputDir = strcat('Data/PCA_final/task',num2str(task),'/fast/');
else
    outputDir = strcat('Data/PCA_final/task',num2str(task),'/slow/');
end
% Check if the folder exists , and if not, make it...
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'result_PCA.mat'),'AccT1','AccT2','AccBoth'...
    ,'ErrI_T1','ErrII_T1'...
    ,'ErrI_T2','ErrII_T2'...
    ,'ErrI_both','ErrII_both'...
    ,'ErrII_T1_test','ErrII_T2_test','ErrII_both_test'...
    ,'X_all','Y_all','X_T1','Y_T1','X_T2','Y_T2');





