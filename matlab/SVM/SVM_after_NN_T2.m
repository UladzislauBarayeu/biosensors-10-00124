function [ mean_result ] = SVM_after_NN_T2( task, KernelSVM, List_of_subject, nn, Size_of_feat, fast, channel_type)
%SVM_AFTER_NN_T2 analyse data after NN
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
Group_size=5;
    
for subject_i=1:size(List_of_subject,2)
    subject=List_of_subject{subject_i};
    dat=loadjson(strcat('Data/NN_convoluted/',channel_type,'/',nn,'/task',num2str(task),'/data_for_svm_s',subject,'.json'));

    nbFolds = size(dat.T2.train_sample,1);

    for f_valid = 1 : nbFolds
        result_accuracy=zeros(Size_of_feat,1);
        Indexes={};
        trainTrials_fold = dat.T2.train_sample{f_valid};
        trainCues_fold = dat.train_y{f_valid}(:,1);
        

    
        nbFolds = 5;
        uniqueTrials = (1:size(trainTrials_fold,1));
        nbTrials = numel(uniqueTrials);            
        foldSize = fix(nbTrials / nbFolds);
        folds = (0 : foldSize : nbTrials);
        folds(nbFolds+1)=nbTrials;
        %% learn for first feature
        accuracy=zeros(1,size(dat.T2.train_sample{1},2));
        for Nchanel=1:size(dat.T2.train_sample{1},2)
            % Cross validate nbFolds
            ACC=zeros(1,nbFolds);
            for f = 1 : nbFolds
                testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));

                trainTrials = trainTrials_fold(~testMask,Nchanel);
                testTrials = trainTrials_fold(testMask,Nchanel);
                testCues = trainCues_fold(testMask,1);
                trainCues = trainCues_fold(~testMask,1);

                % SVM Classifier
                if fast
                    SVMModel = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                            'KernelFunction',KernelSVM,'KernelScale','auto');
                else
                    opts = struct('Optimizer','bayesopt','ShowPlots',false,...
                        'AcquisitionFunctionName','expected-improvement-plus');
                    SVMModel = fitcsvm(trainTrials, trainCues,...
                        'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
                end
                predict = SVMModel.predict(testTrials);
                ACC(f) = mean(predict == testCues);
            end
            accuracy(1,Nchanel)=mean(ACC);
        end
        [sorted,index]=sort(accuracy(1,:));
        for i=1:Group_size
            groups{i}=[];
            groups{i}(1)= index(end-i+1);
        end
        Indexes{1}=groups;
        result_accuracy(1)=sorted(end);
        max_val=sorted(end);


        %% learn all others
        for Nofeat=2:Size_of_feat
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
                            testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));

                            trainTrials = trainTrials_fold(~testMask,FIdx);
                            testTrials = trainTrials_fold(testMask,FIdx);
                            testCues = trainCues_fold(testMask,1);
                            trainCues = trainCues_fold(~testMask,1);

                            % SVM Classifier
                            if fast
                                SVMModel = fitcsvm(trainTrials,trainCues,'Standardize',true,...
                                    'KernelFunction',KernelSVM,'KernelScale','auto');
                            else

                                opts = struct('Optimizer','bayesopt','ShowPlots',false,...
                                    'AcquisitionFunctionName','expected-improvement-plus');
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
            %add best one
            [max_val,~]=max(accuracy(:));
            [~,isort]=sort(accuracy(:));
            for Ngroup=1:Group_size
                ind1=mod(isort(end-Ngroup+1)-1,Group_size)+1;
                ind2=(isort(end-Ngroup+1)-ind1)/Group_size+1;
                groups2{Ngroup}=[groups{ind1} ind2];
            end
            clear groups
            groups=groups2;
            clear groups2

            Indexes{Nofeat}=groups;
            result_accuracy(Nofeat)=max_val;

            if max_val==1 && Nofeat>5
                break
            end
        end
        %% save
        if fast
            outputDir = strcat('Data/NN_results/',channel_type,'/',nn,'/task',num2str(task),'/fast/T2/');
        else
            outputDir = strcat('Data/NN_results/',channel_type,'/',nn,'/task',num2str(task),'/slow/T2/');
        end
        % Check if the folder exists , and if not, make it...
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        save(strcat(outputDir,subject,'_',num2str(f_valid),'.mat'),'Indexes','result_accuracy');


        Max_values(subject_i,f_valid)=max(result_accuracy);
        clearvars -except List_of_subject task KernelSVM fast Size_of_feat ...
            Max_values nn channel_type Group_size...
            nbFolds f_valid dat subject subject_i
    end
end
mean_result=mean(mean(Max_values));

end

