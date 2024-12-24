'''
This file has dataset specific functions to load and preprocess the data from
each dataset into a standard format (per subject numpy arrays). Preprocessing
consists of: loading the raw data, dealing with nan, and resampling to 25Hz
'''

import pandas as pd
import os
import numpy as np
import re
from scipy.signal import resample


def preprocess_DSADS(dataset_dir: str) -> None:
    """ Loads the DSADS raw data and saves it in a standard format.

    https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities

    Each subject's data and labels will be saved as data_[subject].npy and labels_[subject].npy
    in dataset_dir/preprocessed/. The data will have shape (N x C) where N is the number
    of raw samples per subject and C is the number of sensor channels.

    Parameters
    ----------

    dataset_dir: str
        global path of where the dataset has been installed.


    Returns
    -------

    dataset_info: dict
        metadata about the raw data, specifically the 
        sensor channel map, list of subjects, and label map
    """
    
    # DSADS directory structure is a01/p1/s01.txt (activity, subject, segment)
    activity_folders = os.listdir(dataset_dir)

    # Filter folder names that match the structure 'a' followed by exactly two digits
    activity_folders = [folder for folder in activity_folders if re.match(r'^a\d{2}$', folder)]
    activity_folders.sort(key=lambda f: int(re.sub('\D', '', f)))

    subject_folders = os.listdir(os.path.join(dataset_dir,activity_folders[0]))
    subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
    NUM_SUBJECTS = len(subject_folders)

    segment_files = os.listdir(os.path.join(dataset_dir,activity_folders[0],subject_folders[0]))
    segment_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    NUM_SEGMENTS = len(segment_files)
    SEGMENT_LEN,NUM_CHANNELS = pd.read_csv(os.path.join(dataset_dir,activity_folders[0],subject_folders[0],segment_files[0]),header=None).values.shape
    num_samples_per_activity = NUM_SEGMENTS*SEGMENT_LEN

    # we separate the data by subject
    training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
    training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels

    # merge data for each participant into a numpy array
    for subject_i, subject_folder in enumerate(subject_folders):
        for activity_i, activity_folder in enumerate(activity_folders):
            # create the data array which contains samples across all segment files
            data_array = np.zeros((num_samples_per_activity,NUM_CHANNELS))
            label_array = np.zeros(num_samples_per_activity)
            for segment_i, segment_file in enumerate(segment_files):
                data_file_path = os.path.join(dataset_dir,activity_folder,subject_folder,segment_file)
                data_segment = pd.read_csv(data_file_path,header=None).values
                start = segment_i*SEGMENT_LEN
                end = start + SEGMENT_LEN
                data_array[start:end,:] = data_segment[:,:]
                label_array[start:end] = activity_i

            # put into list
            training_data[subject_i].append(data_array)
            training_labels[subject_i].append(label_array)
        
    # now concatenate and save data
    folder = f"{dataset_dir}/preprocessed_data"
    os.mkdir(folder)
    for subject_i in range(NUM_SUBJECTS):
        training_data[subject_i] = np.concatenate(training_data[subject_i])
        training_labels[subject_i] = np.concatenate(training_labels[subject_i])

        np.save(f"{folder}/data_{subject_i+1}",training_data[subject_i])
        np.save(f"{folder}/labels_{subject_i+1}",training_labels[subject_i])


    # ------------- dataset metadata -------------
    body_parts = ['torso','right_arm','left_arm','right_leg','left_leg']
    sensors = ['acc','gyro','mag']
    sensor_dims = 3 # XYZ
    channels_per_sensor = len(sensors)*sensor_dims

    # dict to get index of sensor channel by bp and sensor
    sensor_channel_map = {
        bp: 
        {
            sensor: np.arange(bp_i*channels_per_sensor+sensor_i*sensor_dims,
                            bp_i*channels_per_sensor+sensor_i*sensor_dims+sensor_dims)
                    for sensor_i,sensor in enumerate(sensors)
        } for bp_i,bp in enumerate(body_parts)
    }

    label_map = {
            0:'sitting',
            1:'standing',
            2:'lying on back',
            3:'lying on right side',
            4:'ascending stairs',
            5:'descending stairs',
            6:'standing in elevator',
            7:'moving in elevator',
            8:'walking in parking lot',
            9:'walking on flat treadmill',
            10:'walking on inclined treadmill',
            11:'running on treadmill,',
            12:'exercising on stepper',
            13:'exercising on cross trainer',
            14:'cycling on exercise bike horizontal',
            15:'cycling on exercise bike vertical',
            16:'rowing',
            17:'jumping',
            18:'playing basketball'
            }
    
    dataset_info = {
        'sensor channel map': sensor_channel_map,
        'list of subjects': [subject_i+1 for subject_i in range(NUM_SUBJECTS)],
        'label map': label_map
    }

    return dataset_info


def preprocess_RWHAR():
    pass

def preprocess_PAMAP2():
    pass

def preprocess_Opportunity():
    pass