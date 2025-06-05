# BCI Project by Lotte Michels &amp; Selma Ancel
This Github repository contains the code for the BCI project by Lotte Michels &amp; Selma Ancel.

## Setup
The following files are included:
- *BCI Pre-Processing*: Pre-processing of the raw EEG OpenMIIR files, replicating original work by Stober (2017) that can be found [here](https://github.com/sstober/openmiir/tree/master/eeg/preprocessing/notebooks).
- **: Training the hvEEGNet variational auto-encoder on the pre-processed OpenMIIR epochs, using the network library created by Cisotto et al. (2022) that can be found [here](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/hvEEGNet_paper).
- *BCI hvEEGNet + Machine Learning*: Building and tuning Logistic Regression and SVM classifiers on the hvEEGNet encoded features.
- **: Building and tuning the Conformer model, designed and shared by Song et al. (2022) [here](https://github.com/eeyhsong/EEG-Conformer/tree/main).

## Data
All the data files used for this project have been stored in this [Kaggle dataset](https://www.kaggle.com/datasets/lottemi/openmiir/data). 

## References


