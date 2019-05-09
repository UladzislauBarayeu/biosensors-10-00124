clear all;






%% set
size_of_subject=105;
task=1;
fast=1;
List_of_subject={'s05','s15','s25','s35','s45','s55','s65','s75','s85','s95'};
best_features_T1=zeros(40,10);
best_features_T2=zeros(40,10);

%%
for subject_i=1:size(List_of_subject,2)
    subject=List_of_subject{subject_i};
    if fast
        resT1=load(strcat('Data/SVM_results/task',num2str(task),'/fast/T1/',subject,'.mat'));
    else
        resT1=load(strcat('Data/SVM_results/task',num2str(task),'/slow/T1/',subject,'.mat'));
    end
    [val_T1,ind_max_T1]=max(resT1.result_accuracy);
    if fast
        resT2=load(strcat('Data/SVM_results/task',num2str(task),'/fast/T2/',subject,'.mat'));
    else
        resT2=load(strcat('Data/SVM_results/task',num2str(task),'/slow/T2/',subject,'.mat'));
    end
    [val_T2,ind_max_T2]=max(resT2.result_accuracy);
    if val_T1==1
        groupT1=resT1.Indexes{ind_max_T1}{1};
    else
        groupT1=resT1.Indexes{5}{1};
    end
    for i=1:length(groupT1)
        feat=rem(groupT1(i)-1,40)+1;
        best_features_T1(feat,i)=best_features_T1(feat,i)+1;
    end
    if val_T2==1
        groupT2=resT2.Indexes{ind_max_T2}{1};
    else
        groupT2=resT2.Indexes{5}{1};
    end
    
    for i=1:length(groupT2)
        feat=rem(groupT2(i)-1,40)+1;
        best_features_T2(feat,i)=best_features_T2(feat,i)+1;
    end
    
    
end

for i=1:size(best_features_T1,1)
    T1(i)=sum(best_features_T1(i,:));
    T2(i)=sum(best_features_T2(i,:));
end

[~, ind1]=sort(T1);
[~, ind2]=sort(T2);
res_featT1_all=ind1(end:-1:1);
res_featT2_all=ind2(end:-1:1);
res_featT1=res_featT1_all(1:10);
res_featT2=res_featT2_all(1:10);

data=loadjson(strcat('Data/SVM/data_for_svm_',List_of_subject{1},'.json'));
for i=1:length(data.result_label)/2
    result_labels{i}=data.result_label{i*2};
end
for i=1:length(res_featT1)
    labels_best_T1{i}=result_labels{res_featT1(i)};
    labels_best_T2{i}=result_labels{res_featT2(i)};
end
if fast
    outputDir = strcat('Data/SVM_final/task',num2str(task),'/fast/');
else
    outputDir = strcat('Data/SVM_final/task',num2str(task),'/slow/');
end
% Check if the folder exists , and if not, make it...
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'best_feat_SVM.mat'),'labels_best_T1','labels_best_T2');


% y=[1 40];x=[1 10];
% imagesc(x,y, best_features_T1, [0 max(max(best_features_T1))]);
% title('best features');
% set(gca,'YDir','normal')
% colormap(gca,jet);
% c=colorbar;
% c.Label.String = 'number met';
% 
% imagesc(x,y, best_features_T2, [0 max(max(best_features_T2))]);
% title('best features');
% set(gca,'YDir','normal')
% colormap(gca,jet);
% c=colorbar;
% c.Label.String = 'number met';




