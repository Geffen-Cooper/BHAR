import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from tqdm import tqdm

from models.model_builder import model_builder, sparse_model_builder
from datasets.dataset import HARClassifierDataset, load_har_classifier_dataloaders, generate_activity_sequence
from datasets.preprocess_raw_data import preprocess_DSADS, preprocess_RWHAR, preprocess_PAMAP2
from experiments.train import train, validate
from utils.setup_funcs import PROJECT_ROOT, init_logger, init_seeds
from utils.parse_results import get_results
from energy_harvesting.energy_harvest import EnergyHarvester
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor
from datasets.sparse_data_utils import SparseHarDataset

def get_args():
	parser = argparse.ArgumentParser(
			description="Dataset and model arguments for training HAR policies",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
	parser.add_argument("--eval", action="store_true", help="Get results of pretrained policies")
	parser.add_argument("--checkpoint_prefix", default="logfile", type=str, help="name for classifier training session")
	parser.add_argument("--logging_prefix", default="logfile", type=str, help="name for policy training session")
	parser.add_argument(
			"--policy",
			default="unconstrained_1",
			type=str,
			help="Energy Spending Policy",
		)
	parser.add_argument(
			"--architecture",
			default="attend",
			type=str,
			choices=["attend", "tinyhar", "convlstm"],
			help="HAR architecture",
		)
	parser.add_argument(
			"--dataset",
			default="dsads",
			type=str,
			choices=["dsads", "rwhar", "pamap2", "opportunity"],
			help="HAR dataset",
		)
	parser.add_argument("--seed", default=0, type=int, help="seed for experiment")
	parser.add_argument("--dataset_top_dir", default="~/Projects/data/dsads", type=str, help="path to dataset")
	parser.add_argument('--subjects', 
						default=[1,2,3,4,5,6,7], nargs='+', type=int, help='List of subjects')
	parser.add_argument('--sensors', 
						default=["acc"], nargs='+', type=str, help='List of sensors')
	parser.add_argument('--body_parts',
						default=["torso","right_arm","left_arm","right_leg","left_leg"], nargs='+', type=str, help='List of body parts')
	parser.add_argument('--activities', 
						default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], nargs='+', type=int, help='List of activities')
	parser.add_argument("--val_frac", default=0.1, type=float, help="fraction of training data for validation")

	parser.add_argument("--window_size", default=8, type=int, help="sliding window size for pretrained model")
	parser.add_argument("--overlap_frac", default=0.5, type=float, help="fraction of window to overlap")
	parser.add_argument("--harvesting_sensor_window_size", default=8, type=int, help="window size for KEH sensor")

	parser.add_argument("--leakage", default=6.6e-6, type=float, help="idle power in J")
	parser.add_argument("--sampling_frequency", default=25, type=int, help="acceleromter sampling frequency")
	parser.add_argument("--max_energy", default=200e-6, type=float, help="energy capacity of sensor in J")

	parser.add_argument(
				"--model_type",
				default="dense_synchronous_baseline",
				type=str,
				choices=["dense_synchronous_baseline",
			             "sparse_asychronous_baseline", 
						 "sparse_asychronous_contextualized"],
				help="Sparse Model Type",
			)	
	# parser.add_argument("--batch_size", default=128, type=int, help="training batch size")
	# parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
	# parser.add_argument("--epochs", default=50, type=int, help="training epochs")
	# parser.add_argument("--ese", default=10, type=int, help="early stopping epochs")
	# parser.add_argument("--log_freq", default=200, type=int, help="after how many batches to log")


	args = parser.parse_args()

	return args


def train_LOOCV(**kwargs):
	""" Trains N policies for Leave One Subject Out Cross Validation

		Parameters
		----------
		
		**kwargs:
			parameters used for the dataset (e.g. batch_size, body parts, subjects, etc.)

		Returns
		-------
	"""

	subjects = kwargs['subjects']
	logging_prefix = kwargs['logging_prefix']
	seed = kwargs['seed']

	# setup the session
	logger = init_logger(f"{logging_prefix}/train_log_seed{seed}")
	init_seeds(seed)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	results_table = {subject: None for subject in subjects}

	for subject_i, subject in enumerate(subjects):
		train_subjects = subjects[:subject_i] + subjects[subject_i+1:]
		test_subjects = [subjects[subject_i]]

		logger.info(f"Train Group: {train_subjects} --> Test Group: {test_subjects}")

		# create the dataset
		preprocessed_path = os.path.join(kwargs['dataset_top_dir'], "preprocessed_data")
		if not os.path.isdir(preprocessed_path):
			if kwargs['dataset'] == 'dsads':
				preprocess_DSADS(kwargs['dataset_top_dir'])
			elif kwargs['dataset'] == 'rwhar':
				preprocess_RWHAR(kwargs['dataset_top_dir'])
			elif kwargs['dataset'] == 'pamap2':
				preprocess_PAMAP2(kwargs['dataset_top_dir'])
		kwargs['dataset_dir'] = preprocessed_path

		kwargs['subjects'] = train_subjects
		kwargs['normalize'] = False
		train_ds = HARClassifierDataset(**kwargs,train=True,val=False)
		val_ds = HARClassifierDataset(**kwargs,train=False,val=True)
		kwargs['subjects'] = test_subjects
		test_ds = HARClassifierDataset(**kwargs,train=False,val=False)

		# merge data across subjects for policy training
		#TODO: do not normalize data before sparsifying
		# apply mean and std after
		train_data = np.concatenate(list(train_ds.raw_data.values()))
		train_labels = np.concatenate(list(train_ds.raw_labels.values()))

		val_data = np.concatenate(list(val_ds.raw_data.values()))
		val_labels = np.concatenate(list(val_ds.raw_labels.values()))

		test_data = np.concatenate(list(test_ds.raw_data.values()))
		test_labels = np.concatenate(list(test_ds.raw_labels.values()))

		sensor_channel_map = train_ds.dataset_info['sensor_channel_map']
		active_channels = train_ds.active_channels
		# -------------------

		# first generate the activity sequence (train, val, test)
		min_dur = 10
		max_dur = 30
		train_data_sequence, train_label_sequence = generate_activity_sequence(train_data,train_labels,min_dur,max_dur,25)
		val_data_sequence, val_label_sequence = generate_activity_sequence(val_data,val_labels,min_dur,max_dur,25)
		test_data_sequence, test_label_sequence = generate_activity_sequence(test_data,test_labels,min_dur,max_dur,25)

		# normalize data used for classification, unormalized used for energy harvesting
		normalized_test_data_sequence = (test_data_sequence-train_ds.mean)/(train_ds.std + 1e-5)

		# prepare environment
		eh = EnergyHarvester()
		ehs = EnergyHarvestingSensor(eh, 
							   kwargs['harvesting_sensor_window_size'], 
							   kwargs['leakage'], 
							   kwargs['sampling_frequency'],
							   kwargs['max_energy'])

		# next learn the policy (train, val)
		# policy = train_policy(ehs, train_data_sequence, train_label_sequence, val_data_sequence, val_label_sequence)

		# next apply the policy (test)
		packet_idxs = {}
		per_bp_data = {}
		for bp in kwargs['body_parts']:
			bp_channels = np.where(np.isin(active_channels,sensor_channel_map[bp]['acc']))[0]
			per_bp_data[bp] = normalized_test_data_sequence[:,bp_channels]
			packet_idxs[bp] = ehs.sparsify_data(kwargs['policy'], test_data_sequence[:,bp_channels])
			
		sparse_har_dataset = SparseHarDataset(per_bp_data, test_label_sequence, packet_idxs)

		# next load the pretrained classifier
		kwargs['checkpoint_postfix'] = f"{train_subjects}_seed{seed}.pth"
		sparse_model = sparse_model_builder(**kwargs)


		import time
		# next classify sparse data
		preds = np.zeros(len(test_label_sequence))
		rand_initial_pred = np.random.randint(len(np.unique(test_label_sequence)))
		last_pred = rand_initial_pred
		last_packet_idx = 0
		current_packet_idx = 0
		times = np.zeros(3)
		for packet_i in tqdm(range(len(sparse_har_dataset))):
			c1 = time.time()
			packets = sparse_har_dataset[packet_i]
			c2 = time.time()
			times[0] += c2-c1
			for bp,packet in packets.items():
				if packet['age'] == 0: # most recent arrival
					at = packet['arrival_time']
					break
			# predictions hold until this new packet
			current_packet_idx = at
			preds[last_packet_idx:current_packet_idx] = last_pred
			last_packet_idx = current_packet_idx
			c3 = time.time()
			times[1] += c3-c2

			# get new prediction
			last_pred = torch.argmax(sparse_model(packets)).item()
			c4 = time.time()
			times[2] += c4-c3
			# print(times)
		# exit()
		# extend until end
		preds[last_packet_idx:] = last_pred

		acc = (preds == test_label_sequence).mean()
		print(f"Accuracy: {acc}")
		exit()


		#----------------------

		
		
		
		# # load the one with the best validation accuracy and evaluate on test set
		# model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		# test_acc,test_f1,test_loss = validate(model, test_loader, device, kwargs['loss_fn'])
		# logger.info(f"Test F1: {test_f1}, Test Acc: {test_acc}")
		# logger.info("==========================================\n\n")
		# results_table[subject] = (test_f1, test_acc)


	logger.info(f"Results: {results_table}")
	# create parent directories if needed
	kwargs['train_logname'] = os.path.join(logging_prefix,f"results_seed{seed}")
	path_items = kwargs['train_logname'].split("/")
	if  len(path_items) > 1:
		Path(os.path.join(PROJECT_ROOT,"saved_data/results",*path_items[:-1])).mkdir(parents=True, exist_ok=True)

	with open(os.path.join(PROJECT_ROOT,"saved_data/results",kwargs['train_logname']+".pickle"), 'wb') as file:
		pickle.dump(results_table, file)

if __name__ == '__main__':

	args = get_args()
	eval_only = args.eval
	args = vars(args)

	# organize logs by dataset and architecture
	args['checkpoint_prefix'] = os.path.join(args['dataset'],args['architecture'],args['checkpoint_prefix'])
	args['logging_prefix'] = os.path.join(args['dataset'],args['architecture'],args['logging_prefix'])

	if eval_only:
		base_path = os.path.join(PROJECT_ROOT,"saved_data/results",args['logging_prefix'])
		result_logs = [os.path.join(base_path, filename) for filename in os.listdir(base_path)]
		mean, std = get_results(result_logs)
		print(f"Mean: {round(mean*100,3)}, std: {round(std*100,3)}")
	else:
		train_LOOCV(**args)
	
	