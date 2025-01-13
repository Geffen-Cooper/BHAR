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
import pickle


def find_closest_index(original_array: np.ndarray , new_array: np.ndarray) -> np.ndarray:
	""" When resampling data to a lower sampling rate, the sample level labels also need to be adjusted.
		During resampling, we will get fewer samples so the corresponding time stamps get adjusted.
		To adjust the sample level labels, this function gets the index at the closest time stamp
		in the original array.

	Parameters
	----------

	original_array: np.ndarray
		an array of time stamps for each sample in the orignal data

	new_array: np.ndarray
		an array of time stamps for each sample in the new data (will be shorter if lower sampling rate)


	Returns
	-------

	closest_indices: np.ndarray
		the indices in the orignal label array to access to set the labels for the new sampling rate
	"""

	# find indices of orignal where samples of new array should be inserted to preserve order
	indices = np.searchsorted(original_array, new_array)
	indices = np.clip(indices, 1, len(original_array)-1)
	
	# consider indices 0->n-1 and 1->n
	left_values = original_array[indices - 1]
	right_values = original_array[indices]
	
	# get index with closest timestamp (new array timestamps fall between old array timestamps so need to choose closest)
	closest_indices = np.where(np.abs(new_array - left_values) < np.abs(new_array - right_values),
							   indices - 1,
							   indices)
	
	return closest_indices

def preprocess_DSADS(dataset_dir: str) -> dict:
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

	# merge data for each subject into a numpy array
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
	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)
	for subject_i in range(NUM_SUBJECTS):
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])


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
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': [subject_i+1 for subject_i in range(NUM_SUBJECTS)],
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)


def preprocess_RWHAR(dataset_dir: str) -> dict:
	""" Loads the RWHAR raw data and saves it in a standard format.

	http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset

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

	label_map = {
			0:'climbingdown',
			1:'climbingup',
			2:'jumping',
			3:'lying',
			4:'running',
			5:'sitting',
			6:'standing',
			7:'walking'
			}

	body_parts = ['chest',
			  'forearm',
			  'head',
			  'shin',
			  'thigh',
			  'upperarm',
			  'waist']

	og_sampling_rate = 50
	new_sampling_rate = 25

	activities = label_map.values()

	# RWHAR directory structure is proband1/data/acc_jumping_head.csv
	subject_folders = os.listdir(dataset_dir)

	# Filter folder names that match the structure 'proband' followed by exactly two digits
	subject_folders = [folder for folder in subject_folders if 'proband' in folder]
	subject_folders.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_folders)

	file_count = {sf:0 for sf in subject_folders}
	active_subjects = []

	# first filter out subjects which have missing data
	for subject_i,subject_folder in enumerate(subject_folders):
		# for a given subject, get all activity csvs (8 activities, 7 body parts)
		activity_csvs = os.listdir(os.path.join(dataset_dir,subject_folder,'data'))
		activity_csvs.sort()
		print(subject_folder)

		# keep a dict of activity files present
		act_bp_dict = {act:{} for act in activities}
		for k in act_bp_dict.keys():
			for bp in body_parts:
				act_bp_dict[k][bp] = False

		# iterate over activity files
		for activity_csv in activity_csvs:
			if activity_csv.endswith(".csv"):
				# determine the label and body part
				activity_str = activity_csv.split("_")[1]
				body_part = activity_csv.split("_")[2]
				if body_part.isdigit(): # some files are split into parts
					body_part = activity_csv.split("_")[3]
				file_count[subject_folder] += 1
				body_part = body_part[:-4]
				act_bp_dict[activity_str][body_part] = True

		# if have less than 8*7 True values, then data is missing
		count = 0
		for act in act_bp_dict.keys():
			for bp in act_bp_dict[act]:
				if act_bp_dict[act][bp] == True:
					count += 1
				else:
					print(f"subject {subject_folder} is missing {act}-{bp}")
		if count == len(body_parts)*len(activities):
			active_subjects.append(subject_i)
	print(f"active_subjects (idxs): {active_subjects}")

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels


	# then load all the data
	for subject_i,subject_folder in enumerate(subject_folders):
		if subject_i not in active_subjects:
			print(f"Skipping {subject_folder}")
			continue

		# keep a dict of activity data present
		act_bp_dict = {act:{} for act in activities}
		for k in act_bp_dict.keys():
			for bp in body_parts:
				act_bp_dict[k][bp] = []

		# for a given subject, get all activity csvs (8 activities, 7 body parts)
		activity_csvs = os.listdir(os.path.join(dataset_dir,subject_folder,'data'))
		activity_csvs.sort()
		print(subject_folder)

		# for each activity, get all body part csvs
		for activity_i,activity in enumerate(activities):
			prefix = f"acc_{activity}_"
			for activity_csv in activity_csvs:
				if activity_csv.startswith(prefix) and activity_csv.endswith(".csv"):
					activity_str = activity_csv.split("_")[1]
					body_part = activity_csv.split("_")[2]
					if body_part.isdigit(): # some files are split into parts
						body_part = activity_csv.split("_")[3]
					# load the data for every body part
					file_path = os.path.join(dataset_dir,subject_folder,'data',activity_csv)
					print(f"{activity_str}-{body_part[:-4]}")
					# filter start and end segments with no activity, don't need id from csv
					if activity_str == 'jumping':
						act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[100:,1:])
					else:
						act_bp_dict[activity_str][body_part[:-4]].append(pd.read_csv(file_path).values[3*100:-3*100,1:])

		# first merge csvs for activities that got split into segments
		for act in act_bp_dict.keys():
			for bp in act_bp_dict[act]:
				if len(act_bp_dict[act][bp]) > 1:
					data_arrays = act_bp_dict[act][bp]
					act_bp_dict[act][bp] = np.concatenate(data_arrays,axis=0)
				else:
					# no more list
					act_bp_dict[act][bp] = act_bp_dict[act][bp][0]

		# now try to temporally align data across body parts as best as possible
		for act_i,act in enumerate(act_bp_dict.keys()):
			start_times = []
			for bp in act_bp_dict[act]:
				start_times.append(act_bp_dict[act][bp][0,0])
			print(f"{subject_folder}-{act}: {start_times}")
			latest_start = max(start_times)
			# remove initial rows if can get closer to the latest start time
			for bp in act_bp_dict[act]:
				times = act_bp_dict[act][bp][:,0]
				closest_idx = np.argmin(abs(times - latest_start))
				act_bp_dict[act][bp] = act_bp_dict[act][bp][closest_idx:,1:]
			# get min length so duration is the same
			lengths = []
			for bp in act_bp_dict[act]:
				lengths.append(act_bp_dict[act][bp].shape[0])
			print(f"{subject_folder}-{act}: {lengths}")
			shortest = min(lengths)
			for bp in act_bp_dict[act]:
				act_bp_dict[act][bp] = act_bp_dict[act][bp][:shortest,:]

			# now merge body parts into one array
			trunc_len = shortest
			data_array = np.zeros((trunc_len,3*len(body_parts)))
			label_array = np.zeros(trunc_len)

			for bp_i,bp in enumerate(act_bp_dict[act]):
				data_array[:,bp_i*3:(bp_i+1)*3] = act_bp_dict[act][bp][:trunc_len,:]
			label_array[:] = act_i
			print(data_array.shape)

			# resample data
			resampling_factor = new_sampling_rate / og_sampling_rate
			old_length = len(data_array[:,0])
			new_length = int(old_length * resampling_factor)
			data_array = resample(data_array, new_length,axis=0)

			# resample labels
			t_e = old_length/og_sampling_rate
			t_old = np.linspace(0,t_e,old_length)
			t_e = new_length/new_sampling_rate
			t_new = np.linspace(0,t_e,new_length)
			closest_idxs = find_closest_index(t_old,t_new)
			label_array = label_array[closest_idxs]

			# put into list
			training_data[subject_i].append(data_array)
			training_labels[subject_i].append(label_array)

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now concatenate and save data
	for subject_i in range(NUM_SUBJECTS):
		if subject_i not in active_subjects:
			print(f"Skipping {subject_folder}")
			continue
		print(len(training_data[subject_i]))
		training_data[subject_i] = np.concatenate(training_data[subject_i])
		training_labels[subject_i] = np.concatenate(training_labels[subject_i])
	  
		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
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
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.array(active_subjects)+1,
		'label_map': label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)

def preprocess_PAMAP2(dataset_dir: str) -> dict:
	""" Loads the RWHAR raw data and saves it in a standard format.

	http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset

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

	label_map = {
			1:'lying',
			2:'sitting',
			3:'standing',
			4:'walking',
			5:'running',
			6:'cycling',
			7:'nordic walking',
			12:'ascending stairs',
			13:'descending stairs',
			16:'vacuuming',
			17:'ironing',
			24:'rope jumping'
			}

	body_parts = ['hand',
			  'chest',
			  'ankle']
	
	active_columns = np.array([1, # label
							   4,5,6, # hand acc
							   21,22,23, # chest acc
							   38,39,40]) # ankle acc

	og_sampling_rate = 100
	new_sampling_rate = 25

	activities = label_map.values()

	# PAMAP2 directory structure is Protocol/subject101.dat
	subject_files = os.listdir(dataset_dir)

	# Filter folder names that match the structure 'proband' followed by exactly two digits
	subject_files = [file for file in subject_files if 'subject' in file]
	subject_files.sort(key=lambda f: int(re.sub('\D', '', f)))
	NUM_SUBJECTS = len(subject_files)

	# we separate the data by subject
	training_data = {subject: [] for subject in range(NUM_SUBJECTS)} # raw data
	training_labels = {subject: [] for subject in range(NUM_SUBJECTS)} # raw labels


	# then load all the data
	for subject_i,subject_file in enumerate(subject_files):

		data = pd.read_table(os.path.join(dataset_dir,subject_file), header=None, sep='\s+')
		data_array = data.values[:,active_columns[1:]]
		label_array = data.values[:,active_columns[0]]

		# make labels contiguous
		activities = label_map.keys()
		activity_label_map = { 
			class_idx : label_map[activity_idx] for class_idx, activity_idx in enumerate(activities)
		}
		# print(f"Label Map: {activity_label_map}")

		label_swap = {activity_idx : class_idx for class_idx, activity_idx in enumerate(activities)}
		
		# realign labels to class idxs
		for activity_idx in activities:
			idxs_to_swap = (label_array == activity_idx).nonzero()[0]
			label_array[idxs_to_swap] = label_swap[activity_idx]
  

		# resample data
		resampling_factor = new_sampling_rate / og_sampling_rate
		old_length = len(data_array[:,0])
		new_length = int(old_length * resampling_factor)
		data_array = resample(data_array, new_length,axis=0)

		# resample labels
		t_e = old_length/og_sampling_rate
		t_old = np.linspace(0,t_e,old_length)
		t_e = new_length/new_sampling_rate
		t_new = np.linspace(0,t_e,new_length)
		closest_idxs = find_closest_index(t_old,t_new)
		label_array = label_array[closest_idxs]

		# put into list
		training_data[subject_i] = data_array
		training_labels[subject_i] = label_array

	output_folder = os.path.join(dataset_dir,"preprocessed_data")
	os.makedirs(output_folder,exist_ok=True)

	# now save data
	for subject_i in range(NUM_SUBJECTS):
	  
		print(f"==== {subject_i} ====")
		print(training_data[subject_i].shape)
		print(training_labels[subject_i].shape)

		np.save(os.path.join(output_folder,f"data_{subject_i+1}"),training_data[subject_i])
		np.save(os.path.join(output_folder,f"labels_{subject_i+1}"),training_labels[subject_i])
	
	# ------------- dataset metadata -------------
	sensors = ['acc']
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
	
	dataset_info = {
		'sensor_channel_map': sensor_channel_map,
		'list_of_subjects': np.arange(NUM_SUBJECTS)+1,
		'label_map': activity_label_map
	}
	
	with open(os.path.join(output_folder,"metadata.pickle"), 'wb') as file:
		pickle.dump(dataset_info, file)
	

def preprocess_Opportunity():
	pass


if __name__ == '__main__':
	# preprocess_DSADS(os.path.expanduser("~/Projects/data/dsads"))
	# preprocess_RWHAR(os.path.expanduser("~/Projects/data/rwhar"))
	preprocess_PAMAP2(os.path.expanduser("~/Projects/data/pamap2/"))