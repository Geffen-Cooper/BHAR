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
import pickle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib


class HARClassifierDataset(Dataset):
	""" PyTorch dataset class for HAR data. This is used to train the classifiers

	Parameters
	----------

	dataset_dir: str
		global path to the preprocessed data

	subjects: list (int)
		list of subjects to load data for

	sensors: list (str)
		list of sensors to get channel subset from

	body_parts: list (str)
		list of body parts to get sensor channels from

	activities: list (int)
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

	normalize: bool
		whether to normalize the data
	
	**kwargs:
		makes it easier to pass in args without needing to filter
	"""

	def __init__(self, 
			  dataset_dir: str, 
			  subjects: list, 
			  sensors: list, 
			  body_parts:list , 
			  activities: list,
			  train: bool, 
			  val: bool, 
			  val_frac: float, 
			  window_size: int, 
			  overlap_frac: float = 0.5, 
			  normalize: bool = True,
			  **kwargs):

		if train:
			print("========= Building Training Dataset =========")
		elif val:
			print("========= Building Val Dataset =========")
		else:
			print("========= Building Test Dataset =========")
		# load the metadata
		with open(os.path.join(dataset_dir,'metadata.pickle'), 'rb') as handle:
			self.dataset_info = pickle.load(handle)
		self.sensor_channel_map = self.dataset_info['sensor_channel_map']
		label_map = self.dataset_info['label_map']

		# determine which channels to use (keep relative channel order of original data)
		self.active_channels = []
		for sensor in sensors:
			for bp in body_parts:
				self.active_channels.append(self.sensor_channel_map[bp][sensor])
		self.active_channels = np.sort(np.concatenate(self.active_channels))
		print(f"Body Parts: {body_parts}")
		print(f"Sensors:  {sensors}")
		print(f"Active Channels: {self.active_channels}")


		# load the raw data (keep relative order of original labels)
		prefix = f"{dataset_dir}/"
		self.subjects = subjects
		print(f"Subjects: {subjects}")

		self.raw_data = {subject: [] for subject in subjects}
		self.raw_labels = {subject: [] for subject in subjects}

		for subject in self.subjects:
			self.raw_data[subject] = np.load(f"{prefix}data_{subject}.npy")[:,self.active_channels] # (n,ch)
			self.raw_labels[subject] = np.load(f"{prefix}labels_{subject}.npy") # (n)


		# filter out the selected activities and do train-val split
		activities = np.sort(np.array(activities))
		print(f"Activities: {activities}")
		self.selected_activity_label_map = { 
			class_idx : label_map[activity_idx] for class_idx, activity_idx in enumerate(activities)
		}
		print(f"Label Map: {self.selected_activity_label_map}")

		label_swap = {activity_idx : class_idx for class_idx, activity_idx in enumerate(activities)}

		for subject in subjects:
			# remove data and labels we don't want 
			idxs_to_keep = []
			for activity_idx in activities:
				# keep idxs from selected activity
				idxs = (self.raw_labels[subject] == activity_idx).nonzero()[0]
				# if train or val, then segment, otherwise if test keep all
				train_len = int(len(idxs)*(1-val_frac))
				if train == True:
					idxs = idxs[:train_len]
				elif val == True:
					idxs = idxs[train_len:]
				idxs_to_keep.append(idxs)
			# merge across activities
			idxs_to_keep = np.concatenate(idxs_to_keep)
			
			self.raw_data[subject] = self.raw_data[subject][idxs_to_keep,:]
			self.raw_labels[subject] = self.raw_labels[subject][idxs_to_keep]
			
			# realign labels to class idxs
			for activity_idx in activities:
				idxs_to_swap = (self.raw_labels[subject] == activity_idx).nonzero()[0]
				self.raw_labels[subject][idxs_to_swap] = label_swap[activity_idx]
  
		# normalize
		all_data = np.concatenate(list(self.raw_data.values()))
		if train == True:
			self.mean = np.mean(all_data, axis=0)
			self.std = np.std(all_data, axis=0)
			np.save(os.path.join(dataset_dir,"training_data_mean"),self.mean)
			np.save(os.path.join(dataset_dir,"training_data_std"),self.std)
		else:
			self.mean = np.load(os.path.join(dataset_dir,"training_data_mean.npy"))
			self.std = np.load(os.path.join(dataset_dir,"training_data_std.npy"))
		# apply training mean/std to train/val/test data
		if normalize:
			for subject in self.subjects:
				self.raw_data[subject] = (self.raw_data[subject]-self.mean)/(self.std + 1e-5)

		# create windows, for test data we do dense prediction on every sample
		if train or val:
			stride = int(window_size*(1-overlap_frac))
		else:
			stride = 1
		self.window_idxs = {subject: [] for subject in subjects}
		self.window_labels = {subject: [] for subject in subjects}

		for subject in subjects:
			idxs, labels = self.create_windows(self.raw_data[subject],self.raw_labels[subject],window_size,stride)
			self.window_idxs[subject] = idxs
			self.window_labels[subject] = labels

	@staticmethod
	def create_windows(data: np.ndarray, labels: np.ndarray, window_size: int, stride: int):
		""" Partitions the raw data into windows.

		Parameters
		----------

		data: np.ndarray
			data array of dimension (L x C) where L is the time dimension

		labels: np.ndarray
			label array of dimension L where L is the time dimension

		window_size: int
			number of samples per window

		stride: int
			how much overlap per window in units of samples

		Returns
		-------

		window_idxs: np.ndarray
			an (N x 2) array of the starting and ending idx for each window
			where N is the number of windows
		
		window_labels: np.ndarray
			an array of length N of the labels for each window where N is
			the number of windows
		"""

		# form windows
		start_idxs = np.arange(0,data.shape[0]-window_size,stride)
		end_idxs = start_idxs + window_size

		window_labels = labels[end_idxs-1] # last label in window

		return np.stack([start_idxs,end_idxs]).T, window_labels

	def __getitem__(self, idx):
		# index into subject then window
		# e.g. [[0,1000], [1000,2000]], if idx = 1200 then count is 1000 so idx becomes 200
		count = 0
		for subject, subject_windows in self.window_idxs.items():
			count += subject_windows.shape[0]
			if count > idx:
				count -= subject_windows.shape[0]
				break
		
		idx = idx - count

		# get the window idxs
		start,end = self.window_idxs[subject][idx]
		
		# get the data window
		X = self.raw_data[subject][start:end,:]

		# get the label
		Y = self.window_labels[subject][idx]

		# return the sample and the class
		return torch.tensor(X).float(), torch.tensor(Y).long()

	def __len__(self):
		return sum([len(wl) for wl in self.window_labels.values()])

	def visualize_batch(self,body_part,sensor):
		matplotlib.rcParams.update({'font.size': 6})
		idxs = torch.randperm(len(self))[:16]
		fig,ax = plt.subplots(4,4,figsize=(9,5))
		fig.subplots_adjust(wspace=0.6,hspace=1)
		body_part_idxs = self.sensor_channel_map[body_part][sensor]
		for i,idx in enumerate(idxs):
			bp_0 = np.where(self.active_channels == body_part_idxs[0])[0][0]
			bp_1 = np.where(self.active_channels == body_part_idxs[1])[0][0]
			bp_2 = np.where(self.active_channels == body_part_idxs[2])[0][0]
			sensor_data,l = self.__getitem__(idx)
			x = sensor_data[bp_0,:]
			y = sensor_data[bp_1,:]
			z = sensor_data[bp_2,:]

			i_x = i % 4
			i_y = i // 4
			x_ = np.arange(x.shape[0])
			ax[i_y,i_x].plot(x_,x,label='X')
			ax[i_y,i_x].plot(x_,y,label='Y')
			ax[i_y,i_x].plot(x_,z,label='Z')
			ax[i_y,i_x].set_xlabel("Sample #")
			ax[i_y,i_x].set_ylabel("Value")
			ax[i_y,i_x].set_title(self.selected_activity_label_map[int(l)])
			ax[i_y,i_x].grid()
			ax[i_y,i_x].set_ylim([-2.5,2.5])
		plt.savefig("viz.png")


def load_har_classifier_dataloaders(train_subjects, test_subjects, **kwargs):
	""" Creates train, val, and test dataloaders for HAR classifiers.

		Parameters
		----------

		train_subjects: list (int)
			list of subjects for training
		
		test_subjects: list (int)
			list of subjects for testing
		
		**kwargs:
			parameters used for the dataset (e.g. batch_size, body parts, subjects, etc.)

		Returns
		-------

		train_loader: Dataloader
			PyTorch dataloader for training
		val_loader: Dataloader
			PyTorch dataloader for validation
		test_loader: Dataloader
			PyTorch dataloader for testing
		"""

	batch_size = kwargs['batch_size']
	kwargs['subjects'] = train_subjects
	train_ds = HARClassifierDataset(**kwargs,train=True,val=False)
	val_ds = HARClassifierDataset(**kwargs,train=False,val=True)
	kwargs['subjects'] = test_subjects
	test_ds = HARClassifierDataset(**kwargs,train=False,val=False)

	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True,num_workers=4)
	val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)
	test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)

	return train_loader, val_loader, test_loader


def generate_activity_sequence(data,labels,min_duration,max_duration,sampling_rate):#, seed):
	# np.random.seed(seed)
	# first make contiguous segments
	contig_labels = np.zeros_like(labels)
	contig_data = np.zeros_like(data)
	counter = 0
	for act in np.unique(labels):
		idxs = (labels == act).nonzero()[0]
		contig_labels[counter:counter+len(idxs)] = labels[idxs]
		contig_data[counter:counter+len(idxs),:] = data[idxs,:]
		counter += len(idxs)

	activity_idxs = {i : (contig_labels == i).nonzero()[0] for i in np.unique(contig_labels)}
	duration = np.arange(min_duration,max_duration+1)

	X = np.zeros_like(contig_data)
	y = np.zeros_like(contig_labels)

	activity_counters = np.zeros(len(np.unique(contig_labels))) # keeps track of where we are
	remaining_activities = list(np.unique(contig_labels))
	sample_counter = 0

	while len(remaining_activities) > 0:
		# randomly sample an activity
		act = int(np.random.choice(np.array(remaining_activities), 1)[0])

		# randomly sample a duration
		dur = np.random.choice(duration, 1)[0]

		# access this chunk of data and add to sequence
		start = int(activity_counters[act])
		end = int(start + dur*sampling_rate)

		activity_counters[act] += (end-start)

		# check if hit end
		if end >= activity_idxs[act].shape[0]:
			end = int(activity_idxs[act].shape[0])-1
			remaining_activities.remove(act)

		start = activity_idxs[act][start]
		end = activity_idxs[act][end]

		X[sample_counter:sample_counter+end-start,:] = contig_data[start:end,:]
		y[sample_counter:sample_counter+end-start] = contig_labels[start:end]
		sample_counter += (end-start)

	return X,y



if __name__ == '__main__':
	from experiments.train_har_classifier import get_args
	from preprocess_raw_data import preprocess_DSADS

	# get args as dict
	args = get_args()
	args = vars(args)

	# preprocess
	dataset_dir = os.path.expanduser(args['dataset_dir'])

	# visualize preprocessed data
	args['dataset_dir'] = os.path.join(dataset_dir,"preprocessed_data")
	dataset = HARClassifierDataset(**args,train=True,val=False)
	dataset.visualize_batch("right_leg","acc")
	