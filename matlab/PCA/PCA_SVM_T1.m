function [ mean_result ] = PCA_SVM_T1( task, Size_of_feat, KernelSVM,  fast, knn)
%PCA_SVM_T1 analyse data

size_of_sub=105;

for subject_i=1:size_of_sub
    load(strcat('Data/PCA_SVM/task',num2str(task),'/',num2str(subject_i),'.mat'));
    Indexes={};
    result_accuracy=zeros(Size_of_feat,1);
    
    X=Subject.T1;
    Y=Subject.cues;
    resultIDx=[];
    %Find best Nofeat
    accuracy=zeros(1,size(X,2));
    %% learn first feature
    for Nchanel=1:size(X,2)
        if(find(resultIDx==Nchanel))
            continue
        else

            % Cross validate nbFolds
            aqurasy=0;
            nbFolds = 5;
            uniqueTrials = (1:size(X,1));
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
                testTrials = X(testMask, :);
                trainCues = Y(~testMask);
                trainTrials = X(~testMask, :);

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
    [max_val,imax]=max(accuracy(1,:));
    [~,imaxs]=find(accuracy(1,:)==max_val);
    for i=1:size(imaxs,2)
        groups{i}=[];
        groups{i}(1)= imaxs(i);
    end
    Indexes{1}=groups;
    result_accuracy(1)=max_val;
    %resultIDx=[resultIDx,imax];
    %% learn other
    for Nofeat=2:Size_of_feat
        if max_val==1
            break
        end
        %check all features
        accuracy=zeros(size(groups,2),size(X,2));
        for Nchanel=1:size(X,2)
             for Ngroup=1:size(groups,2)
                if(find(groups{Ngroup}==Nchanel))
                    continue
                else

                    % Cross validate nbFolds
                    aqurasy=0;
                    nbFolds = 5;
                    uniqueTrials = (1:size(X,1));
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
                        testTrials = X(testMask, :);
                        trainCues = Y(~testMask);
                        trainTrials = X(~testMask, :);

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
        [max_val,imax]=max(accuracy(:));
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
    %best_indx(:,subject_i)=resultIDx;
    %% save
    if fast
        if knn
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/fast/T1/knn',num2str(knn),'/');
        else
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/fast/T1/');
        end
    else
        if knn
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/slow/T1/knn',num2str(knn),'/');
        else
            outputDir = strcat('Data/PCA_results/task',num2str(task),'/slow/T1/');
        end
    end
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,num2str(subject_i),'.mat'),'Indexes','result_accuracy');
    Max_values(subject_i)=max(result_accuracy);
    clearvars -except Max_values task KernelSVM fast Size_of_feat size_of_sub knn

    
end
mean_result=mean(Max_values);



end

