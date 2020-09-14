function [ROC_pval_PCA_Inception, ROC_pval_PCA_simple, ROC_pval_Inception_simple] = ...
    ROC_all( task, fast,  nn_simple, nn_inseption, channel_type, PCA_channels_type)
%UNTITLED plot ROC curves 
%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================

if fast
    PCA=load(strcat('Data/PCA_final/task',num2str(task),'/',PCA_channels_type,'/fast/result_PCA'));
else
    PCA=load(strcat('Data/PCA_final/task',num2str(task),'/',PCA_channels_type,'/slow/result_PCA'));
end

if fast
    SVM_inseption=load(strcat('Data/NN_final/',channel_type,'/',nn_inseption,'/task',num2str(task),'/fast/result_SVM'));
else
    SVM_inseption=load(strcat('Data/NN_final/',channel_type,'/',nn_inseption,'/task',num2str(task),'/slow/result_SVM'));
end
if fast
    SVM_simple=load(strcat('Data/NN_final/',channel_type,'/',nn_simple,'/task',num2str(task),'/fast/result_SVM'));
else
    SVM_simple=load(strcat('Data/NN_final/',channel_type,'/',nn_simple,'/task',num2str(task),'/slow/result_SVM'));
end
ROC_pval_PCA_Inception=0;
ROC_pval_PCA_simple=0;
ROC_pval_Inception_simple=0;
% ROC_pval_PCA_Inception = signrank(PCA.Y_all,SVM_inseption.Y_all);
% ROC_pval_PCA_simple = signrank(PCA.Y_all,SVM_simple.Y_all);
% ROC_pval_Inception_simple = signrank(SVM_simple.Y_all,SVM_inseption.Y_all);

%plot
plot(PCA.X_all,PCA.Y_all); hold on;
plot(SVM_inseption.X_all,SVM_inseption.Y_all);
plot(SVM_simple.X_all,SVM_simple.Y_all); hold off;
xlim([0 0.1])
lgd =legend('PCA-SVM','Inception-like NN-SVM','VGG-like NN-SVM');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

if strcmp(channel_type,'8_channels')
    title('ROC(zoomed in) for 8 channels')
elseif strcmp(channel_type,'16_channels')
    title('ROC(zoomed in) for 16 channels')
else
    title('ROC(zoomed in) for 64 channels')
end
%save file
outputjpgDir = strcat('figures/ROC/',channel_type,'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')

plot(PCA.X_all,PCA.Y_all); hold on;
plot(SVM_inseption.X_all,SVM_inseption.Y_all);
plot(SVM_simple.X_all,SVM_simple.Y_all); hold off;
lgd=legend('PCA-SVM','Inception-like NN-SVM','VGG-like NN-SVM');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')
if strcmp(channel_type,'8_channels')
    title('ROC for 8 channels')
elseif strcmp(channel_type,'16_channels')
    title('ROC for 16 channels')
else
    title('ROC for 64 channels')
end
%save file
outputjpgDir = strcat('figures/ROC/',channel_type,'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all-big.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')


end












% predicted_T1=[]; predicted_T2=[]; predicted_all=[]; labels=[];
% temp=1;
% for sub=1:length(List_of_subject)
%     subject=List_of_subject{sub};
%     NN{sub}=loadjson(strcat('Data/NN_convoluted/',channel_type,'/',nn,'/Task',num2str(task),'/predicted_data_for_ROC_s',subject,'.json'));
%     for fold=1:size(NN{sub}.T1,1)
%         for trial=1:size(NN{sub}.T1,2)
%             predicted_T1=[predicted_T1 NN{sub}.T1(fold,trial)];
%             predicted_T2=[predicted_T2 NN{sub}.T2(fold,trial)];
%             labels{temp}= NN{sub}.labels{fold}{trial};
%             temp=temp+1;
%             %both
%             [~,more_sceptic]=min([NN{sub}.T1(fold,trial) NN{sub}.T2(fold,trial)]);
%             if more_sceptic==1
%                 predicted_all=[predicted_all NN{sub}.T1(fold,trial)];
%             else
%                 predicted_all=[predicted_all NN{sub}.T2(fold,trial)];
%             end
%         end
%     end
% end
% 
% [NN_res.X_T1,NN_res.Y_T1,~,~] = perfcurve(labels,predicted_T1,'subject');
% [NN_res.X_T2,NN_res.Y_T2,~,~] = perfcurve(labels,predicted_T2,'subject');
% [NN_res.X_all,NN_res.Y_all,~,~] = perfcurve(labels,predicted_all,'subject');
