"""
Training script for hvEEGNet with pre-processed OpenMIIR data.

This script is an adapted version of the original example training script provided by:
@author : Alberto (Jesus) Zancanaro
@organization : University of Padua

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch
import mne

from sklearn.model_selection import train_test_split

from library.dataset import dataset_time as ds_time
from library.model import hvEEGNet
from library.training import train_generic

from library.config import config_training as ct
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Specific parameters to change inside the dictionary

epochs = 80
path_to_save_model = '30_BCI_hvEEGNet_VAE_weights_c2' #'BCI_hvEEGNet_VAE_weights'
epoch_to_save_model = 1

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Gather all data
subjects=['01','04','06','07','09','11','12','13','14']

X = [] # Placeholder for all EEG trial data 
y = [] # Placeholder for all corresponding trial labels (song IDs)

num_event_dict = { # for stimulus condition 2, for condition 1 replace all final 2's with 1's
    0: 12,
    1: 22,
    2: 32,
    3: 42,
    4: 112,
    5: 122,
    6: 132,
    7: 142,
    8: 212,
    9: 222,
    10: 232,
    11: 242
}
event_dict = { # for stimulus condition 2, for condition 1 replace all final 2's with 1's
    "Chim Chim Cheree (lyrics)": 12,
    "Take Me out to the Ballgame (lyrics)": 22,
    "Jingle Bells (lyrics)": 32,
    "Mary Had a Little Lamb (lyrics)": 42,
    "Chim Chim Cheree": 112,
    "Take Me out to the Ballgame": 122,
    "Jingle Bells": 132,
    "Mary Had a Little Lamb": 142,
    "Emperor Waltz": 212,
    "Hedwig's Theme": 222,
    "Imperial March": 232,
    "Eine Kleine Nachtmusik": 242
}

for subj in subjects:
    path = '/Users/lotte/Documents/School/03 Tilburg University Master/Brain-Computer Interfacing/Variational-Autoencoder-for-EEG-analysis/examples/library/{}_preprocessed.fif'.format(subj)
    data = mne.io.read_raw_fif(path, preload=True, verbose=False)

    events = mne.find_events(data, stim_channel='STI 014')
    exclude_codes = [2000, 2001, 1111, 11,13,14,21,23,24,31,33,34,41,43,44,111,113,114,121,123,124,131,133,134,141,143,144,211,213,214,221,223,224,231,233,234,241,243,244] # Only include condition 2 (imagination)
    #exclude_codes = [2000, 2001, 1111, 12,13,14,22,23,24,32,33,34,42,43,44,112,113,114,122,123,124,132,133,134,142,143,144,212,213,214,222,223,224,232,233,234,242,243,244] # Only include condition 1 (perception)
    filtered_events = events[~np.isin(events[:, 2], exclude_codes)]
    
    data_epochs = mne.Epochs(data, filtered_events, tmin=-0.2, tmax=6.9, event_id=event_dict, preload=True)
    epochs_eeg = data_epochs.copy().pick_types(eeg=True, eog=False, stim=False, misc=False)
    
    EEG_data = epochs_eeg.get_data() # shape = (60, 64, 456), should be (60, 1, 64, 450)
    
    mean = np.mean(EEG_data, axis=2, keepdims=True)
    zero_mean_data = EEG_data - mean

    min_val = np.min(zero_mean_data, axis=2, keepdims=True)
    max_val = np.max(zero_mean_data, axis=2, keepdims=True)

    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    EEG_data = 2 * (zero_mean_data - min_val) / range_val - 1
    
    EEG_data = np.expand_dims(EEG_data, axis=1)  # insert a dimension at index 1
    EEG_data = EEG_data[:, :, :, :450]  # crop last dim to 450 (otherwise the network won't work)
    if type(X) == list:
        X = EEG_data
    else:
        X = np.concatenate([X, EEG_data], axis=0)
    # add to full data collection
    
    id_nums = epochs_eeg.events[:,2]  # original numerical event IDs (last column)
    reverse_dict = {v: k for k, v in num_event_dict.items()}
    labels = [reverse_dict.get(num, "Unknown") for num in id_nums] # create a list of mapped numerical labels (0-11)
    y += labels # add to full data collection

y = np.array(y)

print(X.shape, y.shape) # there should be 540 epochs/trials, each with size 64 (chans) x 450 (timepoints)
print(type(X), type(y)) # both should be numpy arrays

print('DATA IS CONCATENATED')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get data and train config
train_data, validation_data, train_label, validation_label = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

# Create channel lists
ch_list = epochs_eeg.ch_names

# Create train and validation dataset
train_dataset = ds_time.EEG_Dataset(train_data, train_label, ch_list)
validation_dataset = ds_time.EEG_Dataset(validation_data, validation_label, ch_list)

# Get training config
train_config = ct.get_config_hierarchical_vEEGNet_training()

# Update train config
train_config['epochs'] = epochs
train_config['path_to_save_model'] = path_to_save_model
train_config['epoch_to_save_model'] = epoch_to_save_model

print('DATA IS PREPARED, START TRAINING')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get model

# Get number of channels and length of time samples
C = train_data.shape[2]
T = train_data.shape[3]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# If the model has also a classifier add the information to training config
train_config['measure_metrics_during_training'] = model_config['use_classifier']
train_config['use_classifier'] = model_config['use_classifier']

# hvEEGNet creation
model = hvEEGNet.hvEEGNet_shallow(model_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataloader, loss function, optimizer and lr_scheduler

# Create dataloader
train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
loader_list             = [train_dataloader, validation_dataloader]

# Declare loss function
# This method return the PyTorch loss function required by the training function.
# The loss function for hvEEGNet is not directy implemented in PyTorch since it is a combination of different losses. So I have to create my own function to combine all the components.
loss_function = train_generic.get_loss_function(model_name = 'hvEEGNet_shallow', config = train_config)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(),
                              lr = train_config['lr'],
                              weight_decay = train_config['optimizer_weight_decay']
                              )

# (OPTIONAL) Setup lr scheduler
if train_config['use_scheduler'] :
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['lr_decay_rate'])
else:
    lr_scheduler = None
    
# Move the model to training device (CPU/GPU)
model.to(train_config['device'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Train the VAE model
model = train_generic.train(model, loss_function, optimizer,
                            loader_list, train_config, lr_scheduler, model_artifact = None)

print('TRAINING COMPLETED, weights saved in:', path_to_save_model)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -







