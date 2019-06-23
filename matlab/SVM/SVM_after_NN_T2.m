function [ mean_result ] = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast)
%SVM_AFTER_NN_T2 analyse data after NN


for subject_i=1:size(List_of_subject,2)
    subject=List_of_subject{subject_i};
    dat=loadjson(strcat('Data/NN_convoluted/nn',num2str(nn),'/data_for_svm_',subject,'.json'));

    %make train model


    result_accuracy=zeros(Size_of_feat,1);
    Indexes={};
    nbFolds = size(dat.T2.train_sample,1);
    
    %% learn for first feature
    accuracy=zeros(1,size(dat.T2.train_sample{1},2));
    for Nchanel=1:size(dat.T2.train_sample{1},2)
        % Cross validate nbFolds
        ACC=zeros(1,nbFolds);
        for f = 1 : nbFolds

            trainTrials = dat.T2.train_sample{f}(:,Nchanel);
            testTrials = dat.T2.test_sample{f}(:,Nchanel);
            testCues = dat.test_y{f}(:,1);
            trainCues = dat.train_y{f}(:,1);

            % SVM Classifier
            if fast
                SVMModel = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                        'KernelFunction',KernelSVM,'KernelScale','auto');
            else
                opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 5);
                SVMModel = fitcsvm(trainTrials, trainCues,...
                    'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
            end
            predict = SVMModel.predict(testTrials);
            ACC(f) = mean(predict == testCues);
        end
        accuracy(1,Nchanel)=mean(ACC);
    end
    [max_val,imax]=max(accuracy(1,:));
    [~,imaxs]=find(accuracy(1,:)==max_val);
    for i=1:size(imaxs,2)
        groups{i}=[];
        groups{i}(1)= imaxs(i);
    end
    Indexes{1}=groups;
    result_accuracy(1)=max_val;
     
    %% learn all others
    for Nofeat=2:Size_of_feat
        if max_val==1
            break
        end
        %check all features
        accuracy=zeros(size(groups,2),size(dat.T2.train_sample{1},2));
        for Nchanel=1:size(dat.T2.train_sample{1},2)
            for Ngroup=1:size(groups,2)
                if(find(groups{Ngroup}==Nchanel))
                    continue
                else
                    FIdx=[groups{Ngroup}, Nchanel];
                    ACC=zeros(1,nbFolds);
                    for f = 1 : nbFolds
                        trainTrials = dat.T2.train_sample{f}(:,FIdx);
                        testTrials = dat.T2.test_sample{f}(:,FIdx);
                        testCues = dat.test_y{f}(:,1);
                        trainCues = dat.train_y{f}(:,1);

                        % SVM Classifier
                        if fast
                            SVMModel = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                                'KernelFunction',KernelSVM,'KernelScale','auto');
                        else
                            opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 5);
                            SVMModel = fitcsvm(trainTrials, trainCues,...
                                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
                        end
                        predict = SVMModel.predict(testTrials);
                        ACC(f) = mean(predict == testCues);
                    end
                    accuracy(Ngroup, Nchanel)=mean(ACC);
                end
            end
        end
        [max_val,~] = max(accuracy(:));
        ind1=[];ind2=[];
        [ind1,ind2]=find(accuracy==max_val);
        if length(ind1)>5
            r=[];
            r = unique([r ; randsample(length(ind1),5)'],'rows');
            ind1=ind1(r);
            ind2=ind2(r);
        end
        for Ngroup=1:length(ind1)
            groups2{Ngroup}=[groups{ind1(Ngroup)} ind2(Ngroup)];
        end
        clear groups
        groups=groups2;
        clear groups2
        
        Indexes{Nofeat}=groups;
        result_accuracy(Nofeat)=max_val;
        
        if max_val==1
            break
        end
    end
    %% save
    if fast
        outputDir = strcat('Data/NN_results/nn',num2str(nn),'/task',num2str(task),'/fast/T2/');
    else
        outputDir = strcat('Data/NN_results/nn',num2str(nn),'/task',num2str(task),'/slow/T2/');
    end
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,subject,'.mat'),'Indexes','result_accuracy');
    Max_values(subject_i)=max(result_accuracy);
    clearvars -except List_of_subject task KernelSVM fast Size_of_feat Max_values nn
end

mean_result=mean(Max_values);


end

