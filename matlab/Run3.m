%==========================================
%Author: Uladzislau Barayeu
%Github: @UladzislauBarayeu
%Email: uladzislau.barayeu@ist.ac.at
%==========================================
clear all


channel_types={'64_channels','8_channels','16_channels'};

task=1;
fast=1;%if 1 run without optimization
for i=1:length(channel_types) 
    channel_type=channel_types{i};
    
    nn_simple=strcat('nn_simple_1_',channel_type);
    nn_inseption=strcat('nn_inception_1_',channel_type);
    PCA_channels_type=channel_type;
    %% far distribution
    [FAR_pval_PCA_Inception, FAR_pval_PCA_simple, FAR_pval_Inception_simple]=...
        FAR_box(task, fast,  nn_inseption, nn_simple, channel_type, PCA_channels_type);


    %%
    [ROC_pval_PCA_Inception, ROC_pval_PCA_simple, ROC_pval_Inception_simple]=...
        ROC_all( task, fast,  nn_simple, nn_inseption, channel_type, PCA_channels_type);

    outputDir=strcat('DATA/pval_end/',channel_type,'/');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    save(strcat(outputDir,'result_SVM.mat'),'FAR_pval_PCA_Inception','FAR_pval_PCA_simple',...
        'FAR_pval_Inception_simple','ROC_pval_PCA_Inception',...
        'ROC_pval_PCA_simple','ROC_pval_Inception_simple')
end

%% PCA  

PCA_8=load(strcat('Data/PCA_final/task',num2str(task),'/8_channels/fast/result_PCA'));
PCA_16=load(strcat('Data/PCA_final/task',num2str(task),'/16_channels/fast/result_PCA'));
PCA_64=load(strcat('Data/PCA_final/task',num2str(task),'/64_channels/fast/result_PCA'));


FAR_pval_PCA8_PCA16 =  signrank(PCA_8.ErrorIITboth_test,PCA_16.ErrorIITboth_test);
FAR_pval_PCA16_PCA64 = signrank(PCA_16.ErrorIITboth_test,PCA_64.ErrorIITboth_test);
FAR_pval_PCA8_PCA64 = signrank(PCA_8.ErrorIITboth_test,PCA_64.ErrorIITboth_test);


box_data=[PCA_8.ErrorIITboth_test PCA_16.ErrorIITboth_test PCA_64.ErrorIITboth_test];
box_label=zeros(length(PCA_8.ErrorIITboth_test)+length(PCA_16.ErrorIITboth_test)+length(PCA_64.ErrorIITboth_test),11);
box_label=char(box_label);
for i=1:length(PCA_8.ErrorIITboth_test)
    box_label(i,:)='8 channels ';
end
for i=1:length(PCA_16.ErrorIITboth_test)
    box_label(i+length(PCA_8.ErrorIITboth_test),:)='16 channels';
end
for i=1:length(PCA_64.ErrorIITboth_test)
    box_label(i+length(PCA_8.ErrorIITboth_test)+length(PCA_16.ErrorIITboth_test),:)='64 channels';
end
boxplot(box_data,box_label)
title('Distribution of the FAR for PCA-SVM')
    
ylabel('FAR')
ylim([0 0.2])
outputjpgDir = strcat('figures/boxplot/PCA/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR-all-false.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')


plot(PCA_8.X_all,PCA_8.Y_all); hold on;
plot(PCA_16.X_all,PCA_16.Y_all);
plot(PCA_64.X_all,PCA_64.Y_all); hold off;
xlim([0 0.1])
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC(zoomed in) for PCA-SVM')

outputjpgDir = strcat('figures/ROC/PCA/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all-zoomed.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')

plot(PCA_8.X_all,PCA_8.Y_all); hold on;
plot(PCA_16.X_all,PCA_16.Y_all);
plot(PCA_64.X_all,PCA_64.Y_all); hold off;
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC for PCA-SVM')

outputjpgDir = strcat('figures/ROC/PCA/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc');

%% SVM simple
channel_type='8_channels';
nn=strcat('nn_simple_1_',channel_type);
SVM_8=load(strcat('Data/NN_final/8_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

channel_type='16_channels';
nn=strcat('nn_simple_1_',channel_type);
SVM_16=load(strcat('Data/NN_final/16_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

channel_type='64_channels';
nn=strcat('nn_simple_1_',channel_type);
SVM_64=load(strcat('Data/NN_final/64_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

FAR_pval_simple8_simple16 =  signrank(SVM_8.ErrorIITboth_test,SVM_16.ErrorIITboth_test);
FAR_pval_simple16_simple64 = signrank(SVM_16.ErrorIITboth_test,SVM_64.ErrorIITboth_test);
FAR_pval_simple8_simple64 = signrank(SVM_8.ErrorIITboth_test,SVM_64.ErrorIITboth_test);

box_data=[SVM_8.ErrorIITboth_test SVM_16.ErrorIITboth_test SVM_64.ErrorIITboth_test];
box_label=zeros(length(SVM_8.ErrorIITboth_test)+length(SVM_16.ErrorIITboth_test)+length(SVM_64.ErrorIITboth_test),11);
box_label=char(box_label);
for i=1:length(SVM_8.ErrorIITboth_test)
    box_label(i,:)='8 channels ';
end
for i=1:length(SVM_16.ErrorIITboth_test)
    box_label(i+length(SVM_8.ErrorIITboth_test),:)='16 channels';
end
for i=1:length(SVM_64.ErrorIITboth_test)
    box_label(i+length(SVM_8.ErrorIITboth_test)+length(SVM_16.ErrorIITboth_test),:)='64 channels';
end
boxplot(box_data,box_label)
title('Distribution of the FAR for VGG-like NN-SVM')
    
ylabel('FAR')
ylim([0 0.2])
outputjpgDir = strcat('figures/boxplot/SVM_simple/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR-all-false.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')


plot(SVM_8.X_all,SVM_8.Y_all); hold on;
plot(SVM_16.X_all,SVM_16.Y_all);
plot(SVM_64.X_all,SVM_64.Y_all); hold off;
xlim([0 0.1])
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC(zoomed in) for VGG-like NN-SVM')

outputjpgDir = strcat('figures/ROC/SVM_simple/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all-zoomed.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')

plot(SVM_8.X_all,SVM_8.Y_all); hold on;
plot(SVM_16.X_all,SVM_16.Y_all);
plot(SVM_64.X_all,SVM_64.Y_all); hold off;
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC for VGG-like NN-SVM')

outputjpgDir = strcat('figures/ROC/SVM_simple/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')

%% SVM inseption

channel_type='8_channels';
nn=strcat('nn_inception_1_',channel_type);
SVM_8=load(strcat('Data/NN_final/8_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

channel_type='16_channels';
nn=strcat('nn_inception_1_',channel_type);
SVM_16=load(strcat('Data/NN_final/16_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

channel_type='64_channels';
nn=strcat('nn_inception_1_',channel_type);
SVM_64=load(strcat('Data/NN_final/64_channels/',nn,'/task',num2str(task),'/fast/result_SVM'));

FAR_pval_inseption8_inseption16 =  signrank(SVM_8.ErrorIITboth_test,SVM_16.ErrorIITboth_test);
FAR_pval_inseption16_inseption64 = signrank(SVM_16.ErrorIITboth_test,SVM_64.ErrorIITboth_test);
FAR_pval_inseption8_inseption64 = signrank(SVM_8.ErrorIITboth_test,SVM_64.ErrorIITboth_test);

box_data=[SVM_8.ErrorIITboth_test SVM_16.ErrorIITboth_test SVM_64.ErrorIITboth_test];
box_label=zeros(length(SVM_8.ErrorIITboth_test)+length(SVM_16.ErrorIITboth_test)+length(SVM_64.ErrorIITboth_test),11);
box_label=char(box_label);
for i=1:length(SVM_8.ErrorIITboth_test)
    box_label(i,:)='8 channels ';
end
for i=1:length(SVM_16.ErrorIITboth_test)
    box_label(i+length(SVM_8.ErrorIITboth_test),:)='16 channels';
end
for i=1:length(SVM_64.ErrorIITboth_test)
    box_label(i+length(SVM_8.ErrorIITboth_test)+length(SVM_16.ErrorIITboth_test),:)='64 channels';
end
boxplot(box_data,box_label)
title('Distribution of the FAR for Inception-like NN-SVM')
    
ylabel('FAR')
ylim([0 0.2])
outputjpgDir = strcat('figures/boxplot/SVM_inception/task',num2str(task),'/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-FAR-all-false.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')


plot(SVM_8.X_all,SVM_8.Y_all); hold on;
plot(SVM_16.X_all,SVM_16.Y_all);
plot(SVM_64.X_all,SVM_64.Y_all); hold off;
xlim([0 0.1])
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC(zoomed in) for Inception-like NN-SVM')

outputjpgDir = strcat('figures/ROC/SVM_inception/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all-zoomed.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname);
saveas(gcf,outputjpgname(1:end-4),'epsc')

plot(SVM_8.X_all,SVM_8.Y_all); hold on;
plot(SVM_16.X_all,SVM_16.Y_all);
plot(SVM_64.X_all,SVM_64.Y_all); hold off;
lgd=legend('8','16','64');
lgd.Location ='East';
xlabel('False positive rate') 
ylabel('True positive rate')

title('ROC for Inception-like NN-SVM')

outputjpgDir = strcat('figures/ROC/SVM_inception/');
if ~exist(outputjpgDir, 'dir')
        mkdir(outputjpgDir);
end
namefile=strcat('%s','-ROC-all.jpg');
outputjpgname = sprintf(namefile, outputjpgDir);
saveas(gcf,outputjpgname)
saveas(gcf,outputjpgname(1:end-4),'epsc');


outputDir=strcat('DATA/pval_end/');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

save(strcat(outputDir,'result_over_channel.mat'),...
    'FAR_pval_PCA8_PCA16','FAR_pval_PCA8_PCA64','FAR_pval_PCA16_PCA64',...
    'FAR_pval_simple8_simple16','FAR_pval_simple8_simple64','FAR_pval_simple16_simple64',...
    'FAR_pval_inseption8_inseption16','FAR_pval_inseption8_inseption64','FAR_pval_inseption16_inseption64')





