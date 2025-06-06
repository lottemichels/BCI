"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with support function relative to the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

import wandb

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function to download/load file inside artifact

def add_file_to_artifact(artifact, file_name):
    artifact.add_file(file_name)
    wandb.save(file_name)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Function related to load classification metrics

def update_log_dict_metrics(metrics_dict, log_dict, label):
    # return accuracy, cohen_kappa, sensitivity, specificity, f1
    accuracy = metrics_dict['accuracy']
    cohen_kappa = metrics_dict['cohen_kappa']
    sensitivity = metrics_dict['sensitivity']
    specificity = metrics_dict['specificity']
    f1 = metrics_dict['f1']

    log_dict['accuracy_{}'.format(label)] = accuracy
    log_dict['cohen_kappa_{}'.format(label)] = cohen_kappa
    log_dict['sensitivity_{}'.format(label)] = sensitivity
    log_dict['specificity_{}'.format(label)] = specificity
    log_dict['f1_{}'.format(label)] = f1
