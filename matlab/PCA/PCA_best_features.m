clear all;






%% set
size_of_subject=105;
task=1;
fast=1;
best_features_T1=zeros(40,10);
best_features_T2=zeros(40,10);

%%
for subject_i=1:size_of_subject
    if fast
        resT1=load(strcat('Data/PCA_results/task',num2str(task),'/fast/T1/',num2str(subject_i),'.mat'));
    else
        resT1=load(strcat('Data/PCA_results/task',num2str(task),'/slow/T1/',num2str(subject_i),'.mat'));
    end
    [val_T1,ind_max_T1]=max(resT1.result_accuracy);
    if fast
        resT2=load(strcat('Data/PCA_results/task',num2str(task),'/fast/T2/',num2str(subject_i),'.mat'));
    else
        resT2=load(strcat('Data/PCA_results/task',num2str(task),'/slow/T2/',num2str(subject_i),'.mat'));
    end
    [val_T2,ind_max_T2]=max(resT2.result_accuracy);
    if val_T1==1
        groupT1=resT1.Indexes{ind_max_T1}{1};
    else
        groupT1=resT1.Indexes{10}{1};
    end
    for i=1:length(groupT1)
        feat=rem(groupT1(i)-1,40)+1;
        best_features_T1(feat,i)=best_features_T1(feat,i)+1;
    end
    if val_T2==1
        groupT2=resT2.Indexes{ind_max_T2}{1};
    else
        groupT2=resT2.Indexes{10}{1};
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

data=load(strcat('Data/PCA_processed/task',num2str(task),'/',num2str(1),'.mat'));
for i=1:length(res_featT1)
    labels_best_T1{i}=data.Subject.result_label{res_featT1(i)};
    labels_best_T2{i}=data.Subject.result_label{res_featT2(i)};
end
if fast
    outputDir = strcat('Data/PCA_final/task',num2str(task),'/fast/');
else
    outputDir = strcat('Data/PCA_final/task',num2str(task),'/slow/');
end
% Check if the folder exists , and if not, make it...
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'best_feat_PCA.mat'),'labels_best_T1','labels_best_T2');


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




