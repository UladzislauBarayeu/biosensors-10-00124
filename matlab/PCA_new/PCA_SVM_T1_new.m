function [ mean_result ] = PCA_SVM_T1_new( task, Size_of_feat, KernelSVM,  fast, knn,selected_channels, number_sub_channel)
%PCA_SVM_T1 analyse data

size_of_sub=105;
name_folder='';
for nb_channel=1:size( selected_channels,2)
    name_folder=strcat(name_folder,selected_channels{nb_channel});
end
name_folder=strcat(name_folder,'end');
for subject_i=1:size_of_sub
    load(strcat('Data/PCA_SVM/task',num2str(task),'/',name_folder,'/',num2str(subject_i),'.mat'));
    Indexes={};
    result_accuracy=zeros(Size_of_feat,1);
    
    X_before_pca=Subject.T1;
    Y=Subject.cues;
    
    resultIDx=[];
    %Find best Nofeat
    accuracy=zeros(1,size(X_before_pca,2)*number_sub_channel);
    %% learn first feature
    for Nchanel=1:size(X_before_pca,2)*number_sub_channel
        if(find(resultIDx==Nchanel))
            continue
        else

            % Cross validate nbFolds
            aqurasy=0;
            nbFolds = 5;
            uniqueTrials = (1:size(X_before_pca,1));
            nbTrials = numel(uniqueTrials);            
            assert(mod(nbTrials,nbFolds) == 0);
            foldSize = nbTrials / nbFolds;
            folds = (0 : foldSize : nbTrials);
            %select all possible group
            FIdx=[resultIDx, Nchanel];

            ACC=zeros(1,nbFolds);
            for f = 1 : nbFolds
                testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
                testCues = Y(testMask);
                testTrials = X_before_pca(testMask, :, :);
                trainCues = Y(~testMask);
                trainTrials = X_before_pca(~testMask, :, :);


                trainTrials_pca=zeros(size(trainTrials,1),size(trainTrials,2)*number_sub_channel);
                testTrials_pca=zeros(size(testTrials,1),size(testTrials,2)*number_sub_channel);
                for pca_i=1:size(trainTrials,2)
                    pca_data_train=zeros(size(trainTrials,1),size(trainTrials,3));
                    pca_data_test=zeros(size(testTrials,1),size(testTrials,3));

                    for pca_j=1:size(trainTrials,1)
                        for pca_k=1:size(trainTrials,3)
                            pca_data_train(pca_j,pca_k)=trainTrials(pca_j,pca_i,pca_k);
                        end
                    end
                    [COEFF, SCORE, ~, ~, ~, MU]=pca(pca_data_train);

                    score_test=zeros(size(testTrials,1),size(COEFF,2));
                    for pca_j=1:size(testTrials,1)
                        for pca_k=1:size(testTrials,3)
                            pca_data_test(pca_j,pca_k)=testTrials(pca_j,pca_i,pca_k);
                        end
                        pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MU;
                        score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFF';
                    end
                    for sub_channel_i=1:number_sub_channel
                        trainTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=SCORE(:,sub_channel_i);
                        testTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
                    end
                end
                trainTrials=trainTrials_pca;
                testTrials=testTrials_pca;

                trainTrials = trainTrials(:,FIdx);
                testTrials = testTrials(:,FIdx);

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

                    % Cross validate nbFolds
                    aqurasy=0;
                    nbFolds = 5;
                    uniqueTrials = (1:size(X_before_pca,1));
                    nbTrials = numel(uniqueTrials);            
                    assert(mod(nbTrials,nbFolds) == 0);
                    foldSize = nbTrials / nbFolds;
                    folds = (0 : foldSize : nbTrials);
                    %select all possible group
                    FIdx=[groups{Ngroup}, Nchanel];

                    ACC=zeros(1,nbFolds);
                    for f = 1 : nbFolds
                        testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
                        testCues = Y(testMask);
                        testTrials = X_before_pca(testMask, :, :);
                        trainCues = Y(~testMask);
                        trainTrials = X_before_pca(~testMask, :, :);


                        trainTrials_pca=zeros(size(trainTrials,1),size(trainTrials,2)*number_sub_channel);
                        testTrials_pca=zeros(size(testTrials,1),size(testTrials,2)*number_sub_channel);
                        for pca_i=1:size(trainTrials,2)
                            pca_data_train=zeros(size(trainTrials,1),size(trainTrials,3));
                            pca_data_test=zeros(size(testTrials,1),size(testTrials,3));

                            for pca_j=1:size(trainTrials,1)
                                for pca_k=1:size(trainTrials,3)
                                    pca_data_train(pca_j,pca_k)=trainTrials(pca_j,pca_i,pca_k);
                                end
                            end
                            [COEFF, SCORE, ~, ~, ~, MU]=pca(pca_data_train);

                            score_test=zeros(size(testTrials,1),size(COEFF,2));
                            for pca_j=1:size(testTrials,1)
                                for pca_k=1:size(testTrials,3)
                                    pca_data_test(pca_j,pca_k)=testTrials(pca_j,pca_i,pca_k);
                                end
                                pca_data_test(pca_j,:)=pca_data_test(pca_j,:)-MU;
                                score_test(pca_j,:)=pca_data_test(pca_j,:)/COEFF';
                            end
                            for sub_channel_i=1:number_sub_channel
                                trainTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=SCORE(:,sub_channel_i);
                                testTrials_pca(:,(pca_i-1)*number_sub_channel+sub_channel_i)=score_test(:,sub_channel_i);
                            end
                        end
                        trainTrials=trainTrials_pca;
                        testTrials=testTrials_pca;

                        trainTrials = trainTrials(:,FIdx);
                        testTrials = testTrials(:,FIdx);

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
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/fast/T1/knn',num2str(knn),'/');
        else
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/fast/T1/');
        end
    else
        if knn
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/slow/T1/knn',num2str(knn),'/');
        else
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/',name_folder,'/slow/T1/');
        end
    end
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,num2str(subject_i),'.mat'),'Indexes','result_accuracy');
    Max_values(subject_i)=max(result_accuracy);
    clearvars -except Max_values task KernelSVM fast Size_of_feat size_of_sub knn name_folder number_sub_channel

    
end
mean_result=mean(Max_values);



end

