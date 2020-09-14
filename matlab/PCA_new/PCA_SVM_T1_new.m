function [ mean_result ] = PCA_SVM_T1_new( task, Size_of_feat, KernelSVM, ...
    fast, knn,selected_channels, number_sub_channel, size_of_sub)
%PCA_SVM_T1 analyse data
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

for subject_i=1:size_of_sub
    load(strcat('Data/PCA_SVM/task',num2str(task),'/',name_folder,'/',num2str(subject_i),'.mat'));
    
    nbFolds = 5;
    uniqueTrials = (1:size(Subject.T1,1));
    nbTrials_hold = numel(uniqueTrials);            
    assert(mod(nbTrials_hold,nbFolds) == 0);
    foldSize_hold = nbTrials_hold / nbFolds;
    folds_hold = (0 : foldSize_hold : nbTrials_hold);
    
    X_before_pca=Subject.T1;
    Y=Subject.cues;
    for f_valid = 1 : nbFolds
        %% PCA

        testMask = ismember((1:nbTrials_hold), (folds_hold(f_valid)+1 : folds_hold(f_valid+1)));
        trainCues_folds = Y(~testMask);
        trainTrials = X_before_pca(~testMask, :, :);


        trainTrials_pca=zeros(size(trainTrials,1),size(trainTrials,2)*number_sub_channel);
        for pca_i=1:size(trainTrials,2)
            pca_data_train=zeros(size(trainTrials,1),size(trainTrials,3));

            for pca_j=1:size(trainTrials,1)
                for pca_k=1:size(trainTrials,3)
                    pca_data_train(pca_j,pca_k)=trainTrials(pca_j,pca_i,pca_k);
                end
            end
            [COEFF, SCORE, ~, ~, ~, MU]=pca(pca_data_train);

            for sub_channel_i=1:number_sub_channel
                trainTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=SCORE(:,sub_channel_i);
            end
        end
        clear trainTrials %testTrials
    
        Indexes={};
        result_accuracy=zeros(Size_of_feat,1);

        resultIDx=[];
        %Find best Nofeat
        accuracy=zeros(1,size(trainTrials_pca,2));
        
        nbFolds = 5;
        uniqueTrials = (1:size(trainTrials_pca,1));
        nbTrials = numel(uniqueTrials);            
        foldSize = fix(nbTrials / nbFolds);
        folds = (0 : foldSize : nbTrials);
        folds(nbFolds+1)=nbTrials;
        %% learn first feature
        for Nchanel=1:size(trainTrials_pca,2)
            if(find(resultIDx==Nchanel))
                continue
            else

                %select all possible group
                FIdx=[resultIDx, Nchanel];

                ACC=zeros(1,nbFolds);
                for f = 1 : nbFolds
                    testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
                    trainCues=trainCues_folds(~testMask);
                    testCues=trainCues_folds(testMask);
                    trainTrials = trainTrials_pca(~testMask,FIdx);
                    testTrials = trainTrials_pca(testMask,FIdx);

                    %normalize train data
                    %get max and min for train data
                    trainTrials=trainTrials.';
                    testTrials=testTrials.';
                    for normit=1:size(trainTrials,1)
                        [trainTrials(normit,:),max_train, min_train]=...
                            normalize_me(trainTrials(normit, :));

                        for test_iterator=1:size(testTrials,2)
                            testTrials(normit, test_iterator)=(testTrials(normit,test_iterator)-min_train)...
                                /(max_train-min_train);
                        end
                    end
                    trainTrials=trainTrials.';
                    testTrials=testTrials.';

                    %  Classifier
                    if knn
                        Model = fitcknn(trainTrials,trainCues,'NumNeighbors',knn,'Standardize',1);% num_of neubors?
                    else
                        if fast
                            Model = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                                    'KernelFunction',KernelSVM,'KernelScale','auto');
                            %SVMModel2 = fitPosterior(SVMModel);
                            %[~,score_svm] = predict(SVMModel2,testTrials);
                        else
                            opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 5);
                            Model = fitcsvm(trainTrials, trainCues,...
                                'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
                        end
                    end
                    predict = Model.predict(testTrials);
                    ACC(f) = mean(predict == testCues);               
                end
                accuracy(1,Nchanel)=mean(ACC);

            end
            %Nofeat
            %Nchanel
        end
        %add best one
        [sorted,index]=sort(accuracy(1,:));
        for i=1:5
            groups{i}=[];
            groups{i}(1)= index(end-i+1);
        end
        Indexes{1}=groups;
        result_accuracy(1)=sorted(end);
        max_val=sorted(end);

        %% learn other
        for Nofeat=2:Size_of_feat
            if max_val==1
                break
            end
            %check all features
            accuracy=zeros(size(groups,2),size(X_before_pca,2)*number_sub_channel);
            for Nchanel=1:size(X_before_pca,2)*number_sub_channel
                 for Ngroup=1:size(groups,2)
                    if(find(groups{Ngroup}==Nchanel))
                        continue
                    else

                        %select all possible group
                        FIdx=[groups{Ngroup}, Nchanel];

                        ACC=zeros(1,nbFolds);
                        for f = 1 : nbFolds
                            testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
                            trainCues=trainCues_folds(~testMask);
                            testCues=trainCues_folds(testMask);
                            trainTrials = trainTrials_pca(~testMask,FIdx);
                            testTrials = trainTrials_pca(testMask,FIdx);

                            %normalize train data
                            %get max and min for train data
                            trainTrials=trainTrials.';
                            testTrials=testTrials.';
                            for normit=1:size(trainTrials,1)
                                [trainTrials(normit,:),max_train, min_train]=...
                                    normalize_me(trainTrials(normit, :));

                                for test_iterator=1:size(testTrials,2)
                                    testTrials(normit, test_iterator)=(testTrials(normit,test_iterator)-min_train)...
                                        /(max_train-min_train);
                                end
                            end
                            trainTrials=trainTrials.';
                            testTrials=testTrials.';

                            %  Classifier
                            if knn
                                Model = fitcknn(trainTrials,trainCues,'NumNeighbors',knn,'Standardize',1);% num_of neubors?
                            else
                                if fast
                                    Model = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                                            'KernelFunction',KernelSVM,'KernelScale','auto');
                                else
                                    opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 5);
                                    Model = fitcsvm(trainTrials, trainCues,...
                                        'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
                                end
                            end
                            predict = Model.predict(testTrials);
                            ACC(f) = mean(predict == testCues);               
                        end
                        accuracy(Ngroup, Nchanel)=mean(ACC);
                    end
                end
            end
            %add best one
            [max_val,~]=max(accuracy(:));
            [~,isort]=sort(accuracy(:));
            for Ngroup=1:5
                ind1=mod(isort(end-Ngroup+1)-1,5)+1;
                ind2=(isort(end-Ngroup+1)-ind1)/5+1;
                groups2{Ngroup}=[groups{ind1} ind2];
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
        %best_indx(:,subject_i)=resultIDx;
        %% save
        if fast
            if knn
                outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/fast/T1/knn',num2str(knn),'/',KernelSVM,'/');
            else
                outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/fast/T1/',KernelSVM,'/');
            end
        else
            if knn
                outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/slow/T1/knn',num2str(knn),'/',KernelSVM,'/');
            else
                outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/slow/T1/',KernelSVM,'/');
            end
        end
        % Check if the folder exists , and if not, make it...
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        save(strcat(outputDir,num2str(subject_i),'_',num2str(f_valid),'.mat'),'Indexes','result_accuracy');

        Max_values(subject_i,f_valid)=max(result_accuracy);
        clearvars -except Max_values task KernelSVM fast Size_of_feat size_of_sub knn name_folder number_sub_channel ...
            nbTrials_hold folds_hold Subject subject_i X_before_pca Y
    end
    

    
end
mean_result=mean(mean(Max_values));



end

