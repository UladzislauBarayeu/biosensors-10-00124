
%% calculate cross-corelation between channels, and return mean for all subjects
clear all;
task=1;%set number of task
size_of_subjects=105;%set


%%
load(char(strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(1),'.mat')));
R=zeros(size(Subject.result_label,2),size(Subject.result_label,1),size(Subject.result_label,1));
P=zeros(size(Subject.result_label,2),size(Subject.result_label,1),size(Subject.result_label,1));
for Number_of_feat=1:size(Subject.result_label,2)
    matrix=[];
    for Number_of_subject=1:size_of_subjects

        name_file=strcat('Data/Processed/Combined/task',num2str(task),'/',num2str(Number_of_subject),'.mat');
        load(char(name_file));
        for Number_of_trail=1:size(Subject.T1,2)%set
            matrix=[matrix Subject.T1{Number_of_trail}(:,Number_of_feat)];
        end
    end
    [R(Number_of_feat,:,:),P(Number_of_feat,:,:)] = corrcoef(matrix');
end
Result_r=mean(R);
Std_r=std(R);
Result_R=zeros(size(Result_r,2),size(Result_r,3));
Std_R=zeros(size(Result_r,2),size(Result_r,3));
for ch1=1:size(Result_r,2)
    for ch2=1:size(Std_r,3)
        Result_R(ch1,ch2)=Result_r(1,ch1,ch2);
        Std_R(ch1,ch2)=Std_r(1,ch1,ch2);
    end
end
outputjpgDir='Data/cross/';
if ~exist(outputjpgDir, 'dir')
    mkdir(outputjpgDir);
end
save('Data/cross/cross','Result_R','Std_R','R');


