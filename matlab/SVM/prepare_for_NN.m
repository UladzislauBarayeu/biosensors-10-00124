clear all;
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
Size_of_subject=105;
task=1;
size_of_vector=220;
for subject_i=1:Size_of_subject
    name_file=strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(subject_i),'.mat');
    Submain=load(char(name_file));

    randarray=randi([1 Size_of_subject],size_of_vector-min([size(Submain.Subject.T1,2) size(Submain.Subject.T2,2)]),1);
    while find( randarray==subject_i)
        randarray=randi([1 Size_of_subject],size_of_vector-size(Submain.Subject.T1,2),1);
    end
    Subject.cues=zeros(size_of_vector,1);
    for i=1:min([size(Submain.Subject.T1,2) size(Submain.Subject.T2,2)])
        Subject.cues(i)=1;
    end
    for rand_sub=1:size(randarray,1)
        Random_choose{rand_sub}=load(strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(randarray(rand_sub)),'.mat'));
        random_trial(rand_sub)=randi([1 min([size(Random_choose{rand_sub}.Subject.T1,2) size(Random_choose{rand_sub}.Subject.T2,2)])],...
            1,1);
    end
    %% T1
    Subject.T1=zeros(size_of_vector,size(Submain.Subject.T1{1},2),size(Submain.Subject.T1{1},1));
    for Number_trial=1:min([size(Submain.Subject.T1,2) size(Submain.Subject.T2,2)])
        Subject.T1(Number_trial,:,:)=Submain.Subject.T1{Number_trial}.';
    end
    for rand_sub=1:size(randarray,1)
        Number_trial=(min([size(Submain.Subject.T1,2) size(Submain.Subject.T2,2)])+rand_sub);
        Subject.T1(Number_trial,:,:)=Random_choose{rand_sub}.Subject.T1{random_trial(rand_sub)}.';
    end
    
    %% T2
    Subject.T2=zeros(size_of_vector,size(Submain.Subject.T2{1},2),size(Submain.Subject.T2{1},1));
    for Number_trial=1:min([size(Submain.Subject.T2,2) size(Submain.Subject.T2,2)])
        Subject.T2(Number_trial,:,:)=Submain.Subject.T2{Number_trial}.';
    end
    for rand_sub=1:size(randarray,1)
        Number_trial=(min([size(Submain.Subject.T2,2) size(Submain.Subject.T2,2)])+rand_sub);
        Subject.T2(Number_trial,:,:)=Random_choose{rand_sub}.Subject.T2{random_trial(rand_sub)}.';
    end
    Subject.result_label=Submain.Subject.result_label;
    %% mix data
    idx=randperm(numel(Subject.cues));
    Subject.cues(1:end)=Subject.cues(idx);
    Subject.T1(1:end,:,:)=Subject.T1(idx,:,:);
    Subject.T2(1:end,:,:)=Subject.T2(idx,:,:);
%     for i=1:size(Subject.T1,2)
%         dat=Subject.T1(:,i,:);
%         dat=reshape(dat,[64*220,1]);
%         [max_val, ~]=max(dat)
%         [min_val, ~]=min(dat)
%     end
    
    nbFolds = 5;
    uniqueTrials = (1:size(Subject.T1,1));
    nbTrials = numel(uniqueTrials);            
    assert(mod(nbTrials,nbFolds) == 0);
    foldSize = nbTrials / nbFolds;
    folds = (0 : foldSize : nbTrials);

    
    for f = 1 : nbFolds
        testMask = ismember((1:nbTrials), (folds(f)+1 : folds(f+1)));
        py.test_y{f} = Subject.cues(testMask);
        py.T1.test_x{f} = Subject.T1(testMask, :, :);
        py.T2.test_x{f} = Subject.T2(testMask, :, :);
        py.train_y{f} = Subject.cues(~testMask);
        py.T1.train_x{f} = Subject.T1(~testMask, :, :);
        py.T2.train_x{f} = Subject.T2(~testMask, :, :);
        %normalize
        %T1

        for features=1:size(py.T1.train_x{f},2)
            [py.T1.train_x{f}(:,features,:),max_train_T1(features), min_train_T1(features)]=...
                normalize_me_python(py.T1.train_x{f}(:,features, :));

            for trial=1:size(py.T1.test_x{f},1)
                for channels=1:size(py.T1.test_x{f},3)
                    py.T1.test_x{f}(trial, features, channels)=(py.T1.test_x{f}(trial, features, channels)-min_train_T1(features))...
                        /(max_train_T1(features)-min_train_T1(features));
                end
            end
        end

        %T2
        for features=1:size(py.T2.train_x{f},2)
            [py.T2.train_x{f}(:,features,:),max_train_T2(features), min_train_T2(features)]=...
                normalize_me_python(py.T2.train_x{f}(:,features, :));

            for trial=1:size(py.T2.test_x{f},1)
                for channels=1:size(py.T2.test_x{f},3)
                    py.T2.test_x{f}(trial, features, channels)=(py.T2.test_x{f}(trial, features, channels)-min_train_T2(features))...
                        /(max_train_T2(features)-min_train_T2(features));
                end
            end
        end
        py.T1.max{f}=max_train_T1;
        py.T1.min{f}=min_train_T1;
        py.T2.max{f}=max_train_T2;
        py.T2.min{f}=min_train_T2;
        %all false
%         py2.T1.all_false{f}=[];py3.T2.all_false{f}=[];
%         for subject_i_test=1:Size_of_subject
%             if subject_i_test==subject_i
%                 continue
%             else
%                 Subtest=load(strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(subject_i_test),'.mat'));
%                 bag_param=110;
%                 test.T1=zeros(bag_param,size(Subtest.Subject.T1{1},2),size(Subtest.Subject.T1{1},1));
%                 for Number_trial=1:bag_param%fix in future
%                     test.T1(Number_trial,:,:)=Subtest.Subject.T1{Number_trial}.';
%                 end
%                 test.T2=zeros(bag_param,size(Subtest.Subject.T2{1},2),size(Subtest.Subject.T2{1},1));
%                 for Number_trial=1:bag_param%fix in future
%                     test.T2(Number_trial,:,:)=Subtest.Subject.T2{Number_trial}.';
%                 end
%                 %normalize
%                 for features=1:size(test.T1,2)
%                     for trial=1:bag_param%fix in future
%                         for channels=1:size(test.T1,3)
%                             test.T1(trial, features, channels)=(test.T1(trial, features, channels)-min_train_T1(features))...
%                                 /(max_train_T1(features)-min_train_T1(features));
%                         end
%                     end
%                 end
%                 for features=1:size(test.T2,2)
%                     for trial=1:bag_param%fix in future
%                         for channels=1:size(test.T1,3)
%                             test.T2(trial, features, channels)=(test.T2(trial, features, channels)-min_train_T2(features))...
%                                 /(max_train_T2(features)-min_train_T2(features));
%                         end
%                     end
%                 end
%                 py2.T1.all_false{f}=cat(1,py2.T1.all_false{f}, test.T1);
%                 py3.T2.all_false{f}=cat(1,py3.T2.all_false{f}, test.T2);
%                 
%             end
%         end
    end
    %hdf5write(strcat(num2str(subject_i),'.h5'), '/Data', py);
    %hdf5write(strcat(num2str(subject_i),'all_false_T1.h5'), '/Data', py2);
    %hdf5write(strcat(num2str(subject_i),'all_false_T2.h5'), '/Data', py3);
    %savejson('data',py,strcat(num2str(subject_i),'.json'));
    %savejson('data',py2,strcat(num2str(subject_i),'all_false_T1.json'));
    %savejson('data',py3,strcat(num2str(subject_i),'all_false_T2.json'));
    jsonStr = jsonencode(py); 
    outputDir = strcat('Data/Result_json_cut/Task',num2str(task),'/');
    % Check if the folder exists , and if not, make it...
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    fid = fopen(strcat(outputDir, num2str(subject_i),'.json'), 'w'); 
    if fid == -1, error('Cannot create JSON file'); end 
    fwrite(fid, jsonStr, 'char'); 
    fclose(fid);
%     
%     jsonStr = jsonencode(py2); 
%     outputDir = strcat('Data/Result_json_cut/Task',num2str(task),'/');
%     % Check if the folder exists , and if not, make it...
%     if ~exist(outputDir, 'dir')
%         mkdir(outputDir);
%     end
%     fid = fopen(strcat(outputDir, num2str(subject_i),'all_false_T1.json'), 'w'); 
%     if fid == -1, error('Cannot create JSON file'); end 
%     fwrite(fid, jsonStr, 'char'); 
%     fclose(fid);
%     
%     jsonStr = jsonencode(py3); 
%     outputDir = strcat('Data/Result_json_cut/Task',num2str(task),'/');
%     % Check if the folder exists , and if not, make it...
%     if ~exist(outputDir, 'dir')
%         mkdir(outputDir);
%     end
%     fid = fopen(strcat(outputDir, num2str(subject_i),'all_false_T2.json'), 'w'); 
%     if fid == -1, error('Cannot create JSON file'); end 
%     fwrite(fid, jsonStr, 'char'); 
%     fclose(fid);
end


















