function [] = NN_check( task, List_of_subject,  fast, nn,  fast_check, KernelSVM, threshold,channel_type)
%NN_CHECK check results for NN+SVM
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
predict_both_posterior_all=[];
labels_all=[];


%%
for subject_i=1:size(List_of_subject,2)
    subject=List_of_subject{subject_i};

    %ind_max_T2=Size_of_feat;
    %make train model
    data=loadjson(strcat('Data/NN_convoluted/',channel_type,'/',nn,'/task',num2str(task),'/data_for_svm_s',subject,'.json'));
    %data=load(strcat('Data/NN_convoluted/task',num2str(task),'/',num2str(subject_i),'.mat'));
    % test
    test=loadjson(strcat('Data/NN_convoluted/',channel_type,'/',nn,'/task',num2str(task),'/predicted_data_for_SVM_all_false_subjects_s',subject,'.json'));
    
    
    nbFolds = 5;
    

    errorIT1=zeros(1,nbFolds);
    errorIIT1=zeros(1,nbFolds);
    errorIT2=zeros(1,nbFolds);
    errorIIT2=zeros(1,nbFolds);
    errorIboth=zeros(1,nbFolds);
    errorIIboth=zeros(1,nbFolds);
    
    errorIIT1_test=zeros(1,nbFolds);
    errorIIT2_test=zeros(1,nbFolds);
    errorIIboth_test=zeros(1,nbFolds);

    for f = 1 : nbFolds
        if fast
            resT1=load(strcat('Data/NN_results/',channel_type,'/',nn,'/task',...
                num2str(task),'/fast/T1/',subject,'_',num2str(f),'.mat'));
        else
            resT1=load(strcat('Data/NN_results/',channel_type,'/',nn,'/task',...
                num2str(task),'/slow/T1/',subject,'_',num2str(f),'.mat'));
        end
        %
        resT1.result_accuracy(1:5)=0;
        [~,ind_max_T1]=max(resT1.result_accuracy);
        %ind_max_T1=Size_of_feat;
        if fast
            resT2=load(strcat('Data/NN_results/',channel_type,'/',nn,'/task',...
                num2str(task),'/fast/T2/',subject,'_',num2str(f),'.mat'));
        else
            resT2=load(strcat('Data/NN_results/',channel_type,'/',nn,'/task',...
                num2str(task),'/slow/T2/',subject,'_',num2str(f),'.mat'));
        end
        resT2.result_accuracy(1:5)=0;
        [~,ind_max_T2]=max(resT2.result_accuracy);
    
        XT1=data.T1.train_sample{f};
        XT2=data.T2.train_sample{f};
        Y=data.train_y{f};
        Y=Y(:,1);
        %% T1
        FIdx_T1=resT1.Indexes{ind_max_T1}{1}; 
        trainCuesT1 = Y;
        testCuesT1 =data.test_y{f}(:,1);
        trainTrialsT1 = XT1(:, FIdx_T1);
        testTrialsT1 = data.T1.test_sample{f}(:,FIdx_T1);

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
            opts = struct('Optimizer','bayesopt','ShowPlots',false,...
                'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',10);
            SVMModelT1 = fitcsvm(trainTrialsT1, trainCuesT1,'Standardize',true,...
                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
        end
        predictT1 = SVMModelT1.predict(testTrialsT1);
        SVMModelT1_posterior = fitPosterior(SVMModelT1);
        [~,predictT1_posterior] = predict(SVMModelT1_posterior,testTrialsT1);
        for i=1:length(predictT1)
            if predictT1_posterior(i,2)>threshold
                predictT1(i)=1;
            else
                predictT1(i)=0;
            end
        end
        %% T2
        FIdx_T2=resT2.Indexes{ind_max_T2}{1}; 
        
        trainCuesT2 = Y;
        testCuesT2=data.test_y{f}(:,1);
        trainTrialsT2 = XT2(:, FIdx_T2);
        testTrialsT2 = data.T2.test_sample{f}(:,FIdx_T2);

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
            opts = struct('Optimizer','bayesopt','ShowPlots',false,...
                'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',10);
            SVMModelT2 = fitcsvm(trainTrialsT2, trainCuesT2,'Standardize',true,...
                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
        end
        predictT2 = SVMModelT2.predict(testTrialsT2);
        
        %posterior
        SVMModelT2_posterior = fitPosterior(SVMModelT2);
        [~,predictT2_posterior] = predict(SVMModelT2_posterior,testTrialsT2);
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
            %posterior both
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
            good=sum(testCuesT2);
            if testCuesT2(i)==1 && predictT2(i)==0
                errorIT2(f)=errorIT2(f)+1/good;
            end
            if testCuesT2(i)==0 && predictT2(i)==1
                errorIIT2(f)=errorIIT2(f)+1/(size(testCuesT2,1)-good);
            end
            
            %both
            if testCuesT1(i)==1 && predict_both(i)==0
                errorIboth(f)=errorIboth(f)+1/good;
            end
            if testCuesT1(i)==0 && predict_both(i)==1
                errorIIboth(f)=errorIIboth(f)+1/(size(testCuesT1,1)-good);
            end
        end
        for i_label=1:length(testCuesT1)
            if testCuesT1(i_label)
                labels{i_label}='subject'; 
            else
                labels{i_label}='others.'; 
            end
            
        end
        ACCT1(f) = mean(predictT1 == testCuesT1);
        ACCT2(f) = mean(predictT2 == testCuesT1);
        ACC(f) = mean(predict_both.' == testCuesT1);
        %% test
        
        XT1_test=test.T1.test_sample{f};
        XT2_test=test.T1.test_sample{f};
        Y_test=test.test_y{f};
        Y_test=Y_test(:,1);
        %% T1
        FIdx_T1=resT1.Indexes{ind_max_T1}{1}; 
        testTrialsT1_test = XT1_test(:,FIdx_T1);

        %normalize train data
        %get max and min for train data
        testTrialsT1_test=testTrialsT1_test.';
        for normit=1:size(testTrialsT1_test,1)
            for test_iterator=1:size(testTrialsT1_test,2)
                testTrialsT1_test(normit, test_iterator)=(testTrialsT1_test(normit,test_iterator)-min_train_T1(normit))...
                    /(max_train_T1(normit)-min_train_T1(normit));
            end
        end
        testTrialsT1_test=testTrialsT1_test.';

        % SVM Classifier
        predictT1_test = SVMModelT1.predict(testTrialsT1_test);
        [~,predictT1_posterior_test] = predict(SVMModelT1_posterior,testTrialsT1_test);
        for i=1:length(predictT1_test)
            if predictT1_posterior_test(i,2)>threshold
                predictT1_test(i)=1;
            else
                predictT1_test(i)=0;
            end
        end
        %% T2
        FIdx_T2=resT2.Indexes{ind_max_T2}{1}; 
        testTrialsT2_test = XT2_test(:,FIdx_T2);

        %normalize train data
        %get max and min for train data
        testTrialsT2_test=testTrialsT2_test.';
        for normit=1:size(testTrialsT2_test,1)
            for test_iterator=1:size(testTrialsT2_test,2)
                testTrialsT2_test(normit, test_iterator)=(testTrialsT2_test(normit,test_iterator)-min_train_T2(normit))...
                    /(max_train_T2(normit)-min_train_T2(normit));
            end
        end
        testTrialsT2_test=testTrialsT2_test.';

        % SVM Classifier
        predictT2_test = SVMModelT2.predict(testTrialsT2_test);
        [~,predictT2_posterior_test] = predict(SVMModelT2_posterior,testTrialsT2_test);
        %set the threshold
        for i=1:length(predictT2_test)
            if predictT2_posterior_test(i,2)>threshold
                predictT2_test(i)=1;
            else
                predictT2_test(i)=0;
            end
        end
        
        predict_both_test_posterior=[];
        for i=1:size(Y_test,1)
            [~,more_sceptic]=min([predictT1_posterior_test(i,2) predictT2_posterior_test(i,2)]);
            if more_sceptic==1
                predict_both_test_posterior(i)=predictT1_posterior_test(i,2);
            else
                predict_both_test_posterior(i)=predictT2_posterior_test(i,2);
            end
                    
                    
            if predictT1_test(i)==1 && predictT2_test(i)==1
                predict_both_test(i)=1;
            else
                predict_both_test(i)=0;
            end

            %T1
            if Y_test(i)==0 && predictT1_test(i)==1
                errorIIT1_test(f)=errorIIT1_test(f)+1/size(Y_test,1);
            end

            %T2
            if Y_test(i)==0 && predictT2_test(i)==1
                errorIIT2_test(f)=errorIIT2_test(f)+1/size(Y_test,1);
            end
            
            %both
            if Y_test(i)==0 && predict_both_test(i)==1
                errorIIboth_test(f)=errorIIboth_test(f)+1/size(Y_test,1);
            end
        end
        ACCT1(f) = mean(predictT1 == testCuesT1);
        ACCT2(f) = mean(predictT2 == testCuesT1);
        ACC(f) = mean(predict_both.' == testCuesT1);
        
        
        %ROC
        for i=1:length(predict_both_posterior)
            if strcmp(labels{i},'others.')
                predict_both_posterior(i)=predict_both_test_posterior(randi([1 length(predict_both_test_posterior)]));
            end
        end
        predict_both_posterior_all=[predict_both_posterior_all predict_both_posterior];
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
    
    
    ErrorIIT1_test(subject_i)=mean(errorIIT1_test);
    ErrorIIT2_test(subject_i)=mean(errorIIT2_test);
    ErrorIITboth_test(subject_i)=mean(errorIIboth_test);
    
    clear data test
end


[X_all,Y_all,~,AUC_all] = perfcurve(labels_all,predict_both_posterior_all,'subject');
plot(X_all,Y_all)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for two action')
%save file
outputjpgDir = strcat('figures/ROC/NN/',channel_type,'/',nn,'/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-both.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);



AccT1=mean(accuracyT1);AccT2=mean(accuracyT2);AccBoth=mean(accuracyBoth);
ErrI_T1=mean(ErrorI_T1);ErrII_T1=mean(ErrorII_T1);
ErrI_T2=mean(ErrorI_T2);ErrII_T2=mean(ErrorII_T2);
ErrI_both=mean(ErrorI_both);ErrII_both=mean(ErrorII_both);

box_data=[ErrorII_T1 ErrorII_T2 ErrorII_both];
box_label=zeros(length(ErrorIIT1_test)*3,4);
box_label=char(box_label);
for i=1:length(ErrorIIT1_test)
    box_label(i,:)='T1  ';
end
for i=1:length(ErrorIIT2_test)
    box_label(i+length(ErrorIIT1_test),:)='T2  ';
end
for i=1:length(ErrorIITboth_test)
    box_label(i+2*length(ErrorIIT1_test),:)='both';
end
boxplot(box_data,box_label)
title('Distribution of the FAR')
ylim([0 0.6])
ylabel('FAR')
outputjpgDir = strcat('figures/boxplot/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);

%% test
ErrII_T1_test=mean(ErrorIIT1_test);ErrII_T2_test=mean(ErrorIIT2_test);ErrII_both_test=mean(ErrorIITboth_test);

box_data=[ErrorIIT1_test ErrorIIT2_test ErrorIITboth_test];
box_label=zeros(length(ErrorIIT1_test)*3,14);
box_label=char(box_label);
for i=1:length(ErrorIIT1_test)
    box_label(i,:)='T1 all false  ';
end
for i=1:length(ErrorIIT2_test)
    box_label(i+length(ErrorIIT1_test),:)='T2 all false  ';
end
for i=1:length(ErrorIITboth_test)
    box_label(i+2*length(ErrorIIT1_test),:)='both all false';
end
boxplot(box_data,box_label)
title('Distribution of the FAR all false')
ylabel('FAR')
ylim([0 0.6])
outputjpgDir = strcat('figures/boxplot/',channel_type,'/',nn,'/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR-all-false.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);


%% save
if fast
    outputDir = strcat('Data/NN_final/',channel_type,'/',nn,'/task',num2str(task),'/fast/');
else
    outputDir = strcat('Data/NN_final/',channel_type,'/',nn,'/task',num2str(task),'/slow/');
end
% Check if the folder exists , and if not, make it...
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'result_SVM.mat'),'AccT1','AccT2','AccBoth','accuracyT1','accuracyT2','accuracyBoth'...
    ,'ErrI_T1','ErrII_T1','ErrorI_T1','ErrorII_T1'...
    ,'ErrI_T2','ErrII_T2','ErrorI_T2','ErrorII_T2'...
    ,'ErrI_both','ErrII_both','ErrorI_both','ErrorII_both'...
    ,'ErrII_T1_test','ErrII_T2_test','ErrII_both_test'...
    ,'ErrorIIT1_test','ErrorIIT2_test','ErrorIITboth_test'...
    ,'X_all','Y_all','AUC_all');



end

