# BCI Project by Lotte Michels &amp; Selma Ancel
This Github repository contains the code for the BCI project by Lotte Michels &amp; Selma Ancel.

## Setup
The following files are included:
- __BCI Pre-Processing__: Pre-processing of the raw EEG OpenMIIR files, replicating original work by Stober (2017) that can be found [here](https://github.com/sstober/openmiir/tree/master/eeg/preprocessing/notebooks).
- __ __: Training the hvEEGNet variational auto-encoder on the pre-processed OpenMIIR epochs, using the network library created by Cisotto et al. (2022) that can be found [here](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/hvEEGNet_paper).
- __BCI hvEEGNet + Machine Learning__: Building and tuning Logistic Regression and SVM classifiers on the hvEEGNet encoded features.
- __ __: Building and tuning the Conformer model, designed and shared by Song et al. (2022) [here](https://github.com/eeyhsong/EEG-Conformer/tree/main).

## Data
All the data files used for this project have been stored in this [Kaggle dataset](https://www.kaggle.com/datasets/lottemi/openmiir/data). 

## References relevant to the code
- Cisotto, G., Zancanaro, A., Zoppis, I. F., & Manzoni, S. L. (2024). hvEEGNet: a novel deep learning model for high-fidelity EEG reconstruction. Frontiers in Neuroinformatics, 18. https://doi.org/10.3389/fninf.2024.1459970
- Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7, 267. https://doi.org/10.3389/fnins.2013.00267
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … others. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825–2830.
- Song, Y., Zheng, Q., Liu, B. and Gao, X. (2023). EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization. IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, doi: 10.1109/TNSRE.2022.3230250.
- Stober, S. (2017). Toward Studying Music Cognition with Information Retrieval Techniques: Lessons Learned from the OpenMIIR Initiative. Frontiers in Psychology, 8. https://doi.org/10.3389/fpsyg.2017.01255. Related code is published here: https://github.com/sstober/openmiir.




