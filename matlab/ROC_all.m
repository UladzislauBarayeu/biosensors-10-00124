%% set
clear all

size_of_subject=105;
task=1;
fast=1;

if fast
    PCA=load(strcat('Data/PCA_final/task',num2str(task),'/fast/result_PCA'));
else
    PCA=load(strcat('Data/PCA_final/task',num2str(task),'/slow/result_PCA'));
end

if fast
    SVM=load(strcat('Data/SVM_final/task',num2str(task),'/fast/result_SVM'));
else
    SVM=load(strcat('Data/SVM_final/task',num2str(task),'/slow/result_SVM'));
end

List_of_subject={'s05','s15','s25','s35','s45','s55','s65','s75','s85','s95'};
predicted_T1=[]; predicted_T2=[]; predicted_all=[]; labels=[];
temp=1;
for sub=1:length(List_of_subject)
    subject=List_of_subject{sub};
    NN{sub}=loadjson(strcat('Data/NN_final/task',num2str(task),'/predicted_data_for_ROC_',subject,'.json'));
    for fold=1:size(NN{sub}.T1,1)
        for trial=1:size(NN{sub}.T1,2)
            predicted_T1=[predicted_T1 NN{sub}.T1(fold,trial)];
            predicted_T2=[predicted_T2 NN{sub}.T2(fold,trial)];
            labels{temp}= NN{sub}.labels{fold}{trial};
            temp=temp+1;
            %both
            [~,more_sceptic]=min([NN{sub}.T1(fold,trial) NN{sub}.T2(fold,trial)]);
            if more_sceptic==1
                predicted_all=[predicted_all NN{sub}.T1(fold,trial)];
            else
                predicted_all=[predicted_all NN{sub}.T2(fold,trial)];
            end
        end
    end
end

[NN_res.X_T1,NN_res.Y_T1,~,~] = perfcurve(labels,predicted_T1,'subject');
[NN_res.X_T2,NN_res.Y_T2,~,~] = perfcurve(labels,predicted_T2,'subject');
[NN_res.X_all,NN_res.Y_all,~,~] = perfcurve(labels,predicted_all,'subject');
%plot
plot(PCA.X_T1,PCA.Y_T1); hold on;
plot(NN_res.X_T1,NN_res.Y_T1);
plot(SVM.X_T1,SVM.Y_T1); hold off;
legend('PCA+SVM','NN','NN+SVM');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for T1')
%save file
outputjpgDir = strcat('figures/ROC/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-T1.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);

plot(PCA.X_T2,PCA.Y_T2); hold on;
plot(NN_res.X_T2,NN_res.Y_T2);
plot(SVM.X_T2,SVM.Y_T2); hold off;
legend('PCA+SVM','NN','NN+SVM');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for T2')
%save file
outputjpgDir = strcat('figures/ROC/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-T2.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);

plot(PCA.X_all,PCA.Y_all); hold on;
plot(NN_res.X_all,NN_res.Y_all);
plot(SVM.X_all,SVM.Y_all); hold off;
legend('PCA+SVM','NN','NN+SVM');
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification for all')
%save file
outputjpgDir = strcat('figures/ROC/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);