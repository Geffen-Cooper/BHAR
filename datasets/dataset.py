'''
This file defines a standardized dataset class that is shared across
all the datasets.
'''

import pandas as pd
import os
import numpy as np
import re
from scipy.signal import resample
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
import os
import numpy as np


class HARDataset(Dataset):
    """ PyTorch dataset class for HAR data. This is used to train the classifiers

    Parameters
    ----------

    dataset_dir: str
        global path to the preprocessed data

    subjects: list (int, 0 to N-1)
        list of subjects to load data for

    sensors: list (str)
        list of sensors to get channel subset from

    body_parts: list (str)
        list of body parts to get sensor channels from

    activities: list (int, 0 to N-1)
        list of activities to load

    train: bool
        whether to get the training data

    val: bool
        whether to get the validation data

    val_frac: float
        fraction of training data to segment for validation

    window_size: int
        number of samples per window

    overlap_frac: float
        sliding window overlap fraction for training data

    dataset_info: dict
        metadata about the dataset (returned by preprocessing function)

    """

    def __init__(self, dataset_dir, subjects, sensors, body_parts, activities, \
                 train, val, val_frac, window_size, overlap_frac, dataset_info):
        
        self.sensor_channel_map = dataset_info['sensor channel map']
        label_map = self.sensor_channel_map['label map']

        # determine which channels to use
        active_channels = []
        for sensor in sensors:
            for bp in body_parts:
                self.active_channels.append(self.sensor_channel_map[bp][sensor])
        self.active_channels = np.sort(np.concatenate(active_channels))

        # load the raw data
        prefix = f"{dataset_dir}/"
        self.subjects = subjects

        self.raw_data = {subject: [] for subject in subjects}
        self.raw_labels = {subject: [] for subject in subjects}

        for subject in enumerate(self.subjects):
            self.raw_data[subject] = np.load(f"{prefix}data_{subject}.npy")[:,self.active_channels] # (n,ch)
            self.raw_labels[subject] = np.load(f"{prefix}labels_{subject}.npy") # (n)

        # filter out the selected activities
        self.selected_activity_label_map = { 
            class_idx : label_map[activity_idx] for class_idx, activity_idx in enumerate(activities)
        }

        label_swap = {activity_idx : class_idx for class_idx, activity_idx in enumerate(activities)}

        for subject in subjects:
            idxs_to_keep = []
            for activity_idx in activities:
                idxs_to_keep.append(self.raw_labels[subject] == activity_idx).nonzero()[0]
            
            self.raw_data[subject] = self.raw_data[subject][idxs_to_keep,:]
            self.raw_labels[subject] = self.raw_labels[subject][idxs_to_keep]
            
            for activity_idx in activities:
                self.raw_labels[(self.raw_labels[subject] == activity_idx).nonzero()[0]] = label_swap[activity_idx]
                
# 6, 3, 9, 1
# 6 -> 0, 3 -> 1