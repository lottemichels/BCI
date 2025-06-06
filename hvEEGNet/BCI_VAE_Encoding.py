"""
Script for encoding pre-processed OpenMIIR data using the encoder from the trained hvEEGNet VAE. 

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch
import mne

from library.model import hvEEGNet
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Gather all data
subjects=['01','04','06','07','09','11','12','13','14']

X = [] # Placeholder for all EEG trial data 
y = [] # Placeholder for all corresponding trial labels (song IDs)

num_event_dict = { # for stimulus condition 2, for condition 1 replace all final 2's with 1's
    0: 11,
    1: 21,
    2: 31,
    3: 41,
    4: 111,
    5: 121,
    6: 131,
    7: 141,
    8: 211,
    9: 221,
    10: 231,
    11: 241
}
event_dict = { # for stimulus condition 2, for condition 1 replace all final 2's with 1's
    "Chim Chim Cheree (lyrics)": 11,
    "Take Me out to the Ballgame (lyrics)": 21,
    "Jingle Bells (lyrics)": 31,
    "Mary Had a Little Lamb (lyrics)": 41,
    "Chim Chim Cheree": 111,
    "Take Me out to the Ballgame": 121,
    "Jingle Bells": 131,
    "Mary Had a Little Lamb": 141,
    "Emperor Waltz": 211,
    "Hedwig's Theme": 221,
    "Imperial March": 231,
    "Eine Kleine Nachtmusik": 241
}

for subj in subjects:
    path = '/Users/lotte/Documents/School/03 Tilburg University Master/Brain-Computer Interfacing/Variational-Autoencoder-for-EEG-analysis/examples/library/{}_preprocessed.fif'.format(subj)
    data = mne.io.read_raw_fif(path, preload=True, verbose=False)

    events = mne.find_events(data, stim_channel='STI 014')
    #exclude_codes = [2000, 2001, 1111, 11,13,14,21,23,24,31,33,34,41,43,44,111,113,114,121,123,124,131,133,134,141,143,144,211,213,214,221,223,224,231,233,234,241,243,244] # Only include condition 2 (imagination)
    exclude_codes = [2000, 2001, 1111, 12,13,14,22,23,24,32,33,34,42,43,44,112,113,114,122,123,124,132,133,134,142,143,144,212,213,214,222,223,224,232,233,234,242,243,244] # Only include condition 1 (perception)
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

print(X.shape, y.shape) # there should be 540 epochs/trials (each 64 chans x 450 timepoints) and 540 corresponding labels
print(type(X), type(y)) # both should be numpy arrays

print('DATA IS PREPARED, EXTRACT ENCODER')


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Use trained VAE to encode our data

# Load the model
model_config = cm.get_config_hierarchical_vEEGNet(64, 450)
model = hvEEGNet.hvEEGNet_shallow(model_config)
weights_path = '/Users/lotte/Documents/School/03 Tilburg University Master/Brain-Computer Interfacing/Variational-Autoencoder-for-EEG-analysis/30_BCI_hvEEGNet_VAE_weights_c1/model_BEST.pth' #BCI_hvEEGNet_VAE_weights
model.load_state_dict(torch.load(weights_path, map_location = torch.device('cpu')))

# Extract encoder
vae = model.h_vae # self.h_vae = hierarchical_VAE.hVAE(encoder_cell_list, decoder_cell_list, config), 'self' refers to our model variable
encoder = vae.encoder # self.encoder = hvae_encoder(encoder_cell_list, config), 'self' refers to our vae variable
encoder.eval() # put in evaluation mode

print('ENCODER IS EXTRACTED, START ENCODING')

# Encode data
X_tensor = torch.tensor(X, dtype=torch.float32)
print(X_tensor.shape)
with torch.no_grad():
    # the encode method encodes a input tensor x, returning [z, mu, sigma, cell_outputs]
    X_encoded = encoder.encode(X_tensor, return_distribution=True) # the encode method encodes a input tensor x, returning [z, mu, sigma, cell_outputs]
    latent_tensor = X_encoded[0] 
    print(latent_tensor.shape)

latent_features = latent_tensor.numpy()  # convert to numpy
np.save('c1_latent_features.npy', latent_features) # save features # latent_features.npy
np.save('c1_labels.npy', y) # save corresponding labels # labels.npy

print('DATA IS ENCODED AND SAVED')





