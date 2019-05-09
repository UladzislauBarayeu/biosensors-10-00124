clear all;




List_of_subject={'s4','s5','s6','s16','s17','s18'};
task=1;

for subject_i=1:size(List_of_subject,2)
    subject=List_of_subject{subject_i};
    dat=loadjson(strcat('Data/SVM/data_for_svm_',subject,'.json'));

    %make train model

    Size_of_feat=10;
    result_accuracy=zeros(Size_of_feat,1);
    nbFolds = size(dat.T2.train_sample,1);
    KernelSVM='linear';%'rbf' or 'linear' optional
    Indexes={};
    
    
    %load prelearned
    outputDir = strcat('Data/SVM_results/task',num2str(task),'/T2/');
    dat2=load(strcat(outputDir,subject,'.mat'));
    for i=1:size(dat2.Indexes,2)
        Indexes{i}=dat2.Indexes{i};
        result_accuracy(i)=dat2.result_accuracy(i);
    end
    groups=Indexes{size(dat2.Indexes,2)};
    if(find(result_accuracy==1))
        break;
    end
    %% learn all others
    for Nofeat=(size(dat2.Indexes,2)+1):Size_of_feat
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

                        opts = struct('ShowPlots',false,'MaxObjectiveEvaluations', 5);
                        SVMModel = fitcsvm(trainTrials, trainCues,...
                            'KernelFunction',KernelSVM,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts);
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
        ind1=ind1.'; ind2=ind2.';
        if size(ind1,1)>5
            r = randi([1 size(ind1,1)],1,5);
            ind1=ind1(r);
            ind2=ind2(r);
        end
        for Ngroup=1:size(ind1,1)
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
    outputDir = strcat('Data/SVM_results/task',num2str(task),'/T2/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    save(strcat(outputDir,subject,'.mat'),'Indexes','result_accuracy');
end







