function [ FAR_pval_PCA_Inception, FAR_pval_PCA_simple, FAR_pval_Inception_simple] =...
    FAR_box(task, fast,  nn_inseption, nn_simple, channel_type, PCA_channels_type)

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

minval=min(length(PCA.ErrorIITboth_test),length(SVM_inseption.ErrorIITboth_test));
FAR_pval_PCA_Inception =  signrank(PCA.ErrorIITboth_test(1:minval),SVM_inseption.ErrorIITboth_test(1:minval));
minval=min(length(PCA.ErrorIITboth_test),length(SVM_simple.ErrorIITboth_test));
FAR_pval_PCA_simple = signrank(PCA.ErrorIITboth_test(1:minval),SVM_simple.ErrorIITboth_test(1:minval));
minval=min(length(SVM_simple.ErrorIITboth_test),length(SVM_inseption.ErrorIITboth_test));
FAR_pval_Inception_simple = signrank(SVM_simple.ErrorIITboth_test(1:minval),SVM_inseption.ErrorIITboth_test(1:minval));

box_data=[PCA.ErrorIITboth_test SVM_inseption.ErrorIITboth_test SVM_simple.ErrorIITboth_test];
box_label=zeros(length(PCA.ErrorIITboth_test)+length(SVM_inseption.ErrorIITboth_test)+length(SVM_simple.ErrorIITboth_test),21);
box_label=char(box_label);
for i=1:length(PCA.ErrorIITboth_test)
    box_label(i,:)='PCA-SVM              ';
end
for i=1:length(SVM_inseption.ErrorIITboth_test)
    box_label(i+length(PCA.ErrorIITboth_test),:)='Inception-like NN-SVM';
end
for i=1:length(SVM_simple.ErrorIITboth_test)
    box_label(i+length(PCA.ErrorIITboth_test)+length(SVM_inseption.ErrorIITboth_test),:)='VGG-like NN-SVM      ';
end
boxplot(box_data,box_label)
if strcmp(channel_type,'8_channels')
    title('Distribution of the FAR for 8 channels')
elseif strcmp(channel_type,'16_channels')
    title('Distribution of the FAR for 16 channels')
else
    title('Distribution of the FAR for 64 channels')
end
    
ylabel('FAR')
ylim([0 0.2])
outputjpgDir = strcat('figures/boxplot/',channel_type,'/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR-all-false.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')
end

