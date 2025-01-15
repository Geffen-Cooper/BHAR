import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score

from models.model_builder import model_builder, sparse_model_builder
from datasets.dataset import HARClassifierDataset, load_har_classifier_dataloaders, generate_activity_sequence
from datasets.preprocess_raw_data import preprocess_DSADS, preprocess_RWHAR, preprocess_PAMAP2
from experiments.train import train, validate
from utils.setup_funcs import PROJECT_ROOT, init_logger, init_seeds
from utils.parse_results import get_results
from energy_harvesting.energy_harvest import EnergyHarvester
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor
from datasets.sparse_data_utils import SparseHarDataset
from experiments.zero_order_algos import signSGD, SGD, PatternSearch
from experiments.ZOTrainer_modified import ZOTrainer
from experiments.sensor_reward import PolicyTrain, reward
import multiprocessing

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
				default="synchronous_multisensor",
				type=str,
				choices=["synchronous_multisensor",
						 "asynchronous_single_sensor", 
						 "asynchronous_multisensor_time_context"],
				help="Sparse Model Type",
			)	
	
	# zero order arguments
	parser.add_argument("--batch_size", default=16, type=int, help="training batch size")
	parser.add_argument("--lr", default=[1e-6,5], nargs='+', type=float, help="learning rate for each policy parameter")
	parser.add_argument("--epochs", default=5, type=int, help="training epochs")
	parser.add_argument("--val_every_epochs", default=1, type=int, help="after how many batches to log")
	parser.add_argument("--param_init_vals", default=[0.0,0.0], nargs='+',type=float, help="init value for each parameter")
	parser.add_argument("--param_min_vals", default=[0.0,0.0], nargs='+',type=float, help="min parameter bound")
	parser.add_argument("--param_max_vals", default=[1.5e-4,10000], nargs='+',type=float, help="max parameter bound")


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
		kwargs['train_logname'] = f"{logging_prefix}/{test_subjects}_seed{seed}"

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
		train_data_sequence, train_label_sequence = generate_activity_sequence(train_data,train_labels,min_dur,max_dur,kwargs['sampling_frequency'])
		val_data_sequence, val_label_sequence = generate_activity_sequence(val_data,val_labels,min_dur,max_dur,kwargs['sampling_frequency'])
		test_data_sequence, test_label_sequence = generate_activity_sequence(test_data,test_labels,min_dur,max_dur,kwargs['sampling_frequency'])

		# normalize data used for classification, unormalized used for energy harvesting
		normalized_train_data_sequence = (train_data_sequence-train_ds.mean)/(train_ds.std + 1e-5)
		normalized_val_data_sequence = (val_data_sequence-train_ds.mean)/(train_ds.std + 1e-5)
		normalized_test_data_sequence = (test_data_sequence-train_ds.mean)/(train_ds.std + 1e-5)

		# prepare environment
		eh = EnergyHarvester()
		ehs = EnergyHarvestingSensor(eh, 
							   kwargs['harvesting_sensor_window_size'], 
							   kwargs['leakage'], 
							   kwargs['sampling_frequency'],
							   kwargs['max_energy'])

		# load trained classifier to use for policy evaluation
		kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
		sparse_model = sparse_model_builder(**kwargs)

		policy = kwargs['policy']

		# ============== next learn the policy (train, val) if we want to
		if 'conservative' in kwargs['policy']:
			logger.info("Train policy ===========")
			# if the policy is already trained, just load parameters
			ckpt_path = os.path.join(PROJECT_ROOT,"saved_data/checkpoints",kwargs['train_logname'])+'.pkl'
			if os.path.exists(ckpt_path):
				logger.info("Policy Already Trained")
				with open(ckpt_path, 'rb') as file:
					policy = pickle.load(file)['best']
				logger.info(f"Policy: {policy}")

			else: # otherwise, train the policy

				train_helper = PolicyTrain(active_channels, ehs, train_data_sequence, normalized_train_data_sequence,
									train_label_sequence, val_data_sequence, normalized_val_data_sequence,
									val_label_sequence,sensor_channel_map,sparse_model,**kwargs)
				
				# create a checkpoint, init sensor policies to opportunistic
				policy = {'current': {bp: [0.,0.] for bp in kwargs['body_parts']},
			  			  'best': {bp: [0.,0.] for bp in kwargs['body_parts']}}
				with open(ckpt_path, 'wb') as file:
					pickle.dump(policy, file)

				file_lock = multiprocessing.Lock()
				barrier = multiprocessing.Barrier(len(kwargs['body_parts']))

				processes = []
				zo_trainers = []

				# optimize sensors iteratively
				for bp in kwargs['body_parts']:
					params_bounds = [[x, y] for x, y in zip(kwargs['param_min_vals'], kwargs['param_max_vals'])]
					optimizer_cfg = {
						'optimizer': PatternSearch, # PatternSearch, signSGD, SGD
						'init_params': kwargs['param_init_vals'],
						'lr': kwargs['lr'],
						'params_bounds': params_bounds,
					}

					train_cfg = {
						'batch_size': kwargs['batch_size'],
						'epochs': kwargs['epochs'],
						'val_every_epochs': kwargs['val_every_epochs'],
						'train_seg_duration':200*kwargs['sampling_frequency']	
					}
					
					# # load the current policy so we can get the frozen params
					# with open(ckpt_path, 'rb') as file:
					# 	frozen_policy = pickle.load(file)
					# frozen_policy.pop(bp)

					logger.info(f"Training: {bp}, policy: {policy['current']}")

					eh = EnergyHarvester()
					ehs = EnergyHarvestingSensor(eh, 
										kwargs['harvesting_sensor_window_size'], 
										kwargs['leakage'], 
										kwargs['sampling_frequency'],
										kwargs['max_energy'])

					train_helper = PolicyTrain(active_channels, ehs, train_data_sequence, normalized_train_data_sequence,
									train_label_sequence, val_data_sequence, normalized_val_data_sequence,
									val_label_sequence,sensor_channel_map,sparse_model,**kwargs)

					# train the policy for the given bp with others frozen
					zo_trainer = ZOTrainer(optimizer_cfg,train_cfg,train_helper,reward,logger,bp, file_lock, barrier)
					zo_trainers.append(zo_trainer)
					
					# zo_trainer.train()

				for i, bp in enumerate(kwargs['body_parts']):
					p = multiprocessing.Process(target=zo_trainers[i].train,args=(i,))
					processes.append(p)
					p.start()

				for p in processes:
					p.join()
					

				# load trained policy
				with open(ckpt_path, 'rb') as file:
					policy = pickle.load(file)['best']
		else:
			# opportunistic or dense for all sensors
			policy = {bp: kwargs['policy'] for bp in kwargs['body_parts']}


		# ============= after getting policy, finetune model if needed
		
		# finetune the model if contextualized specified, otherwise load the other models
		# if the checkpoint already exists then just load it

		# get train, val, test loaders
			# apply the trained policy to get sparse hard dataset objects
		# load contextualized model
		# put in standard training loop using existing train function

		eh = EnergyHarvester()
		ehs = EnergyHarvestingSensor(eh, 
							   kwargs['harvesting_sensor_window_size'], 
							   kwargs['leakage'], 
							   kwargs['sampling_frequency'],
							   kwargs['max_energy'])
			

		# test the learned policy and/or contextualized model
		packet_idxs = {}
		per_bp_data = {}
		for bp in kwargs['body_parts']:
			bp_channels = np.where(np.isin(active_channels,sensor_channel_map[bp]['acc']))[0]
			per_bp_data[bp] = normalized_test_data_sequence[:,bp_channels]
			if 'conservative' in kwargs['policy']:
				packet_idxs[bp] = ehs.sparsify_data(f"conservative_{policy[bp][0]}_{policy[bp][1]}", test_data_sequence[:,bp_channels])
			else:
				packet_idxs[bp] = ehs.sparsify_data(policy[bp], test_data_sequence[:,bp_channels])
		
		sparse_har_dataset = SparseHarDataset(per_bp_data, test_label_sequence, packet_idxs)

		active_idxs, passive_idxs = sparse_har_dataset.region_decomposition()
		# print(f"Active: {len(active_idxs)/(len(active_idxs)+len(passive_idxs))}")
		# print(f"Passive: {len(passive_idxs)/(len(active_idxs)+len(passive_idxs))}")

		# next load the trained classifier
		kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
		sparse_model = sparse_model_builder(**kwargs)
		sparse_model.eval()

		# next classify sparse data
		preds = np.zeros(len(test_label_sequence))
		rand_initial_pred = np.random.randint(len(np.unique(test_label_sequence)))
		last_pred = rand_initial_pred
		last_packet_idx = 0
		current_packet_idx = 0
		for packet_i in tqdm(range(len(sparse_har_dataset))):
			packets, labels = sparse_har_dataset[packet_i]
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
				last_pred = torch.argmax(sparse_model(packets)).item()
		
		# extend until end
		preds[last_packet_idx:] = last_pred

		test_acc = (preds == test_label_sequence).mean()
		test_f1 = f1_score(test_label_sequence,preds,average='macro')
		
		logger.info(f"Test F1: {test_f1}, Test Acc: {test_acc}")
		active_region = len(active_idxs)/(len(active_idxs)+len(passive_idxs))
		active_error = 1-(preds[active_idxs] == test_label_sequence[active_idxs]).mean()
		passive_region = len(passive_idxs)/(len(active_idxs)+len(passive_idxs))
		passive_error = 1-(preds[passive_idxs] == test_label_sequence[passive_idxs]).mean()
		logger.info(f"Active Region: {round(active_region,3)}, Active Error: {round(active_error,3)} --> {round(active_region*active_error,3)} ")
		logger.info(f"Passive Region: {round(passive_region,3)}, Passive Error: {round(passive_error,3)} --> {round(passive_region*passive_error,3)}")
		logger.info("==========================================\n\n")
		results_table[subject] = (test_f1, test_acc)


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
	
	