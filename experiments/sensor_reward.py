import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
from datetime import datetime
import os
from pathlib import Path

from utils.setup_funcs import PROJECT_ROOT
from datasets.sparse_data_utils import SparseHarDataset

# this class should handle all the intermediate steps of getting the reward
# its a class to avoid having to pass in a lot of args to nathan's code
# it handles data formatting, data sampling, applying policy, getting reward
class PolicyTrain():
	def __init__(self,active_channels,harvesting_sensor,
			  	 train_data, train_data_n, train_labels,
				 val_data, val_data_n, val_labels, sensor_channel_map,
				 model, **kwargs):
		self.active_channels = active_channels
		self.harvesting_sensor = harvesting_sensor
		self.train_data = train_data
		self.train_data_n = train_data_n
		self.train_labels = train_labels
		self.val_data = val_data
		self.val_data_n = val_data_n
		self.val_labels = val_labels
		self.model = model

		
		# path for tensorboard
		now = datetime.now()
		now = now.strftime("%Y-%m-%d_%H:%M:%S")
		self.runs_path = os.path.join(PROJECT_ROOT,"saved_data/runs",kwargs['train_logname'])+"_policy_"+now

		
		# first split the data by bp
		self.per_bp_data_train = {}
		self.per_bp_data_val = {}
		self.per_bp_data_train_n = {}
		self.per_bp_data_val_n = {}
		for bp in kwargs['body_parts']:
			bp_channels = np.where(np.isin(active_channels,sensor_channel_map[bp]['acc']))[0]
			self.per_bp_data_train[bp] = self.train_data[:,bp_channels]
			self.per_bp_data_val[bp] = self.val_data[:,bp_channels]
			self.per_bp_data_train_n[bp] = self.train_data_n[:,bp_channels]
			self.per_bp_data_val_n[bp] = self.val_data_n[:,bp_channels]

	def sample_train_segment(self,duration):
		rand_start = np.random.randint(len(self.train_labels))
		# make sure segment doesn't exceed end of data
		if rand_start + duration >= len(self.train_labels):
			rand_start = len(self.train_labels) - duration

		data_seg = {}
		data_seg_n = {}
		for bp in self.per_bp_data_train.keys():
			data_seg[bp] = self.per_bp_data_train[bp][rand_start:rand_start+duration,:]
			data_seg_n[bp] = self.per_bp_data_train_n[bp][rand_start:rand_start+duration,:]
		label_seg = self.train_labels[rand_start:rand_start+duration]

		return data_seg, data_seg_n, label_seg

def reward(latest_params,frozen_sensor_params,per_bp_data,per_body_part_data_normalized,label_sequence,harvesting_sensor,reward_type,classifier,train_mode):

	packet_idxs = {}
	# apply the policy for each sensor using original data
	traces = []
	num_packets = 0
	for bp in per_bp_data.keys():
		# set the policy for frozen sensors
		if bp in frozen_sensor_params.keys():
			policy = f"conservative_{frozen_sensor_params[bp][0]}_{frozen_sensor_params[bp][1]}"
		else: # policy for sensor we are optimizing
			policy = f"conservative_{latest_params[0]}_{latest_params[1]}"
		packet_idxs[bp] = harvesting_sensor.sparsify_data(policy, per_bp_data[bp])
		num_packets += len(packet_idxs[bp])
		traces.append(harvesting_sensor.e_trace)

		# import matplotlib.pyplot as plt
		# # plt.close()
		# ax2 = plt.gca().twinx()
		# ax2.plot(harvesting_sensor.e_trace)
		# plt.savefig("test2.png")
		# plt.scatter(active_idxs,np.zeros_like(active_idxs),c='g',label='active')
		# plt.scatter(passive_idxs,np.zeros_like(passive_idxs),c='r',label='passive')

	# format sparsified data (normalized)
	# import matplotlib.pyplot as plt

	# zero active region
	if num_packets == 0:
		return 0
	sparse_har_dataset = SparseHarDataset(per_body_part_data_normalized, label_sequence, packet_idxs)
	# plt.close()
	# fig,ax = plt.subplots(1,1,figsize=(14,5))
	# ax.plot(label_sequence)
	# # ax.plot(per_bp_data[bp])
	# bps = list(per_bp_data.keys())
	# active_idxs,passive_idxs = sparse_har_dataset.region_decomposition()
	# ax.scatter(active_idxs,np.zeros_like(active_idxs),c='g',label='active')
	# ax.scatter(passive_idxs,np.zeros_like(passive_idxs),c='r',label='passive')
	# # ax.scatter(sparse_har_dataset.unique_packet_timestamps[:,0],np.zeros_like(sparse_har_dataset.unique_packet_timestamps[:,0]),c='k',label='sent')
	# ax.scatter(packet_idxs[bps[0]][:,0],np.zeros_like(packet_idxs[bps[0]][:,0]),c='k',label='sent')
	# print(f"p:{policy}, a: {len(active_idxs)/(len(active_idxs)+len(passive_idxs))}")
	# ax2 = plt.gca().twinx()
	# ax2.plot(traces[0])
	# ax.set_xlim([0,1500])
	# plt.savefig(f'{policy}_{len(active_idxs)/(len(active_idxs)+len(passive_idxs))}.png')

	# get reward from sparsified data
	if reward_type == "active_region":
		active_idxs, passive_idxs = sparse_har_dataset.region_decomposition()
		reward = len(active_idxs)/(len(active_idxs)+len(passive_idxs))
		
	elif reward_type == "accuracy":
		# next classify sparse data
		preds = np.zeros(len(label_sequence))
		# this causes the inconsistency across runs (just predict zeros for consistency)
		# rand_initial_pred = np.random.randint(len(np.unique(label_sequence)))
		rand_initial_pred = 0
		last_pred = rand_initial_pred
		last_packet_idx = 0
		current_packet_idx = 0
		for packet_i in tqdm(range(len(sparse_har_dataset))):
			packets,_ = sparse_har_dataset[packet_i]
			for bp,packet in packets.items():
				if packet['age'] == 0: # most recent arrival
					at = packet['arrival_time']
					break
			# predictions hold until this new packet
			current_packet_idx = at
			
			preds[last_packet_idx:current_packet_idx] = last_pred
			last_packet_idx = current_packet_idx

			# get new prediction
			with torch.no_grad():
				last_pred = torch.argmax(classifier(packets)).item()
		
		# extend until end
		preds[last_packet_idx:] = last_pred

		reward = (label_sequence==preds).mean()
		f1 = f1_score(label_sequence,preds,average='macro')

		if not train_mode:
			reward = (reward,f1)
		# else:
		# 	print(label_sequence)
		# 	print(preds)
		# 	print(reward)
	
	return reward


