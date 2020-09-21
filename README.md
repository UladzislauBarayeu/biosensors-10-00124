# biosensors-10-00124

This repository contains the implementation of Authentication System, described in the paper:
- [Barayeu, U.](https://github.com/UladzislauBarayeu); [Horlava, N.](https://github.com/HorlavaNastassya); Libert, A.; Van Hulle, M. 
[*Robust Single-Trial EEG-Based Authentication Achieved with a 2-Stage Classifier* ](https://www.mdpi.com/2079-6374/10/9/124)
Biosensors 2020, 10, 124.

Please [cite the paper and the code](#how-to-cite) if you are using this project in your research.

## Package dependencies
1. For preprocessing, please install matlab2016 
2. Please download and add the following open source functions/packages to Matlab:
  - [SampEn] (https://de.mathworks.com/matlabcentral/fileexchange/35784-sample-entropy) - calcutates Sample Entropy
  - [EMD] (https://titanwolf.org/Network/Articles/Article?AID=dff3bceb-589d-4768-a306-8de460f60928#gsc.tab=0) - performs Empirical Mode Decomposition
  - [readEDF] (https://nl.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/38641/versions/4/previews/ReadEDF.m/index.html) - reads .edf files
  - [ApEn] (https://la.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/50289/versions/1/previews/Entropy_measures/ApEn.m/index.html) - calculates Aproximate Entropy
  - [jsonlab toolbox v1.5] (https://github.com/flatironinstitute/ironclust/tree/master/matlab/jsonlab-1.5) - reads .json files
  - [Chronux toolbox v2.12] (http://chronux.org) - peforms Power Spectral Density estimation

2. Please install Python 3.7 (or higher), better via Anaconda3
- Install numpy, tensorflow, scikit-learn and keras package via conda:
```conda install numpy
conda install tensorflow
conda install scikit-learn
conda install keras
``` 
## Dataset

- Download data via the link https://archive.physionet.org/pn4/eegmmidb/ and put this data into your path/Data/input folder

## Running the code 

### Data preprocessing: 
- To preprocess the data and perform feature extraction, run 'matlab/Run0.m' script in Matlab
- To create data for NN and to perform PCA+SVM model training,run 'matlab/Run1.m' script in Matlab

### Training Neural Network:
- to create new neural network, cd to Python folder and run the following commands in the Command Prompt:
```cd Python/nn_models
python @name_of_script_for_desired_nn, i.e. nn_inception_1_for_64_channels.py or nn_simple_1_for_8_channels.py 
cd ../ 
python Run_subjects.py --arg2 @name_of_desired_nn  --arg3 nbr_channels --arg5 idx_subject_from --arg6 idx_subject_end
```
For example:
```cd Python/nn_models
python nn_inception_1_for_64_channels.py
cd ../
python Run_subjects.py --arg2 inception_1_64_channels --arg3 64_channels --arg5 0 --arg6 104 
```
### Training SVM for NN+SVM decoder:
- To perform NN+SVM models, run 'matlab/Run2.m' script in Matlab

### Obtaining the results:
- To create ROC figures and box plots, run 'matlab/Run3.m' script in Matlab

## How to cite

Paper:
- Barayeu, U.; Horlava, N.; Libert, A.; Van Hulle, M. Robust Single-Trial EEG-Based Authentication Achieved with a 2-Stage Classifier. Biosensors 2020, 10, 124


