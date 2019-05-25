nrOfLabels = 9;% if you have 9 possible choices

%init
Classifier = SwldaClassifier(1:nrOfLabels);

load(strcat('Data/PCA_SVM/task',num2str(task),'/',num2str(subject_i),'.mat'));
    
X=Subject.T1;
Y=Subject.cues;
% load in:  trainEpochs: the epochs in nb_epochs x samples
%           trainStimuli: labels of which epoch belongs to which target
%           (nb_epochs x 1)
%           trainTarget: labels of which epochs you want it to train
Classifier.train(trainEpochs, trainStimuli, trainTarget, trainTrials);

%load in :  Epochs: the epochs in nb_epochs x samples
%           Stimuli: labels of which epoch belongs to which target
%           (nb_epochs x 1)
predictedTarget = Classifier.classify(epochs, stimuli);