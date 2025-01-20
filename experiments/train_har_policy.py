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
import multiprocessing
import json

from models.model_builder import model_builder, sparse_model_builder
from datasets.dataset import HARClassifierDataset, load_har_classifier_dataloaders, generate_activity_sequence
from datasets.preprocess_raw_data import preprocess_DSADS, preprocess_RWHAR, preprocess_PAMAP2
from experiments.train import train, validate
from utils.setup_funcs import PROJECT_ROOT, MODEL_ROOT, DATA_ROOT, init_logger, init_seeds
from utils.parse_results import get_results
from energy_harvesting.energy_harvest import EnergyHarvester
from energy_harvesting.harvesting_sensor import EnergyHarvestingSensor
from datasets.sparse_data_utils import SparseHarDataset
from experiments.zero_order_algos import signSGD, SGD, PatternSearch
from experiments.ZOTrainer_modified import ZOTrainer
from experiments.sensor_reward import PolicyTrain, reward


def get_args():
	parser = argparse.ArgumentParser(
			description="Dataset and model arguments for training HAR policies",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
	parser.add_argument("--eval", action="store_true", help="Get results of pretrained policies (optional)")
	parser.add_argument("--single_sensor_checkpoint_prefix", default="single_sensor_logfile", type=str, help="name for single sensor classifier training session (optioanl)")
	parser.add_argument("--multisensor_checkpoint_prefix", default="multisensor_logfile", type=str, help="name for multisensor classifier training session (optional)")
	parser.add_argument("--logging_prefix", default="logfile", type=str, help="name for training session (optional)")
	parser.add_argument(
			"--policy",
			default="unconstrained",
			type=str,
			choices=["unconstrained", "opportunistic", "conservative"],
			help="Energy Spending Policy",
			required=True
		)
	parser.add_argument("--unconstrained_stride",default=1,type=int,help="If using unconstrained policy, what stride to use for inference sliding windows (optional)",)
	parser.add_argument(
			"--model_type",
			default="synchronous_multisensor",
			type=str,
			choices=["synchronous_multisensor",
					 "asynchronous_single_sensor", 
					 "asynchronous_multisensor",
					 "asynchronous_multisensor_time_context"],
			help="Sparse Model Type",
		)	
	parser.add_argument(
			"--architecture",
			default="attend",
			type=str,
			choices=["attend", "tinyhar", "convlstm"],
			help="HAR architecture used for single sensor or multisensor classification",
			required=True
		)
	parser.add_argument(
			"--dataset",
			default="dsads",
			type=str,
			choices=["dsads", "rwhar", "pamap2", "opportunity"],
			help="HAR dataset",
			required=True
		)
	parser.add_argument("--seed", default=0, type=int, help="seed for experiment, this must match the seeds used when training the classifier", required=True)
	parser.add_argument('--subjects', default=[1,2,3,4,5,6,7], nargs='+', type=int, help='List of subjects', required = True)
	parser.add_argument('--sensors', default=["acc"], nargs='+', type=str, help='List of sensors', required=True)
	parser.add_argument('--body_parts',default=["torso","right_arm","left_arm","right_leg","left_leg"], nargs='+', type=str, help='List of body parts', required=True)
	parser.add_argument('--activities', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], nargs='+', type=int, help='List of activities',required=True)
	parser.add_argument("--val_frac", default=0.1, type=float, help="fraction of training data for validation",required=True)
	parser.add_argument("--window_size", default=8, type=int, help="sliding window size for pretrained model",required=True)
	parser.add_argument("--harvesting_sensor_window_size", default=8, type=int, help="window size for KEH sensor, for unconstrained policy this is packet size",required=True)

	parser.add_argument("--leakage", default=6.6e-6, type=float, help="idle power in J")
	parser.add_argument("--sampling_frequency", default=25, type=int, help="acceleromter sampling frequency")
	parser.add_argument("--max_energy", default=200e-6, type=float, help="energy capacity of sensor in J")
	
	# zero order arguments
	parser.add_argument("--policy_batch_size", default=16, type=int, help="training batch size")
	parser.add_argument("--policy_lr", default=[1e-6,5], nargs='+', type=float, help="learning rate for each policy parameter")
	parser.add_argument("--policy_epochs", default=5, type=int, help="training epochs")
	parser.add_argument("--policy_val_every_epochs", default=1, type=int, help="after how many batches to log")
	parser.add_argument("--policy_param_init_vals", default=[0.0,0.0], nargs='+',type=float, help="init value for each parameter")
	parser.add_argument("--policy_param_min_vals", default=[0.0,0.0], nargs='+',type=float, help="min parameter bound")
	parser.add_argument("--policy_param_max_vals", default=[1.5e-4,10000], nargs='+',type=float, help="max parameter bound")


	# finetuning arguments
	parser.add_argument("--finetune_batch_size", default=32, type=int, help="finetuning batch size")
	parser.add_argument("--finetune_lr", default=1e-4, type=float, help="learning rate for finetuning model")
	parser.add_argument("--finetune_epochs", default=5, type=int, help="finetuning epochs")

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

	logger.info(json.dumps(kwargs,indent=4))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	kwargs['device'] = device

	results_table = {subject: None for subject in subjects}

	for subject_i, subject in enumerate(subjects):
		train_subjects = subjects[:subject_i] + subjects[subject_i+1:]
		test_subjects = [subjects[subject_i]]

		logger.info(f"Train Group: {train_subjects} --> Test Group: {test_subjects}")

		# generic name of this log
		kwargs['train_logname'] = f"{logging_prefix}/{test_subjects}_seed{seed}"

		# create the dataset
		preprocessed_path = os.path.join(DATA_ROOT[kwargs['dataset']], "preprocessed_data")
		if not os.path.isdir(preprocessed_path):
			if kwargs['dataset'] == 'dsads':
				preprocess_DSADS(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'rwhar':
				preprocess_RWHAR(DATA_ROOT[kwargs['dataset']])
			elif kwargs['dataset'] == 'pamap2':
				preprocess_PAMAP2(DATA_ROOT[kwargs['dataset']])
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
		# the opportunity dataset is collected as sequence of activities so 
		# we don't need to simulate a sequence of activities
		if kwargs['dataset'] != "opportunity":
			min_dur = 10
			max_dur = 30
			np.random.seed(seed)
			train_data_sequence, train_label_sequence = generate_activity_sequence(train_data,train_labels,min_dur,max_dur,kwargs['sampling_frequency'])
			val_data_sequence, val_label_sequence = generate_activity_sequence(val_data,val_labels,min_dur,max_dur,kwargs['sampling_frequency'])
			test_data_sequence, test_label_sequence = generate_activity_sequence(test_data,test_labels,min_dur,max_dur,kwargs['sampling_frequency'])
		else:
			train_data_sequence, train_label_sequence = train_data,train_labels
			val_data_sequence, val_label_sequence = val_data,val_labels
			test_data_sequence, test_label_sequence = test_data,test_labels


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

		policy = kwargs['policy']

		# ================================== Learn or load the policy
		if kwargs['policy'] == 'conservative':
			logger.info("Train policy ===========")

			# load trained classifier to use for policy evaluation
			# this should return the asychronous_single_sensor model since
			# we cannot use the pretrained multisensor models to evaluate the policy
			kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
			model_type = kwargs['model_type']
			kwargs['model_type'] = 'asynchronous_single_sensor'
			sparse_model,_ = sparse_model_builder(**kwargs)

			# if the policy is already trained, just load parameters
			policy_ckpt_path = os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{logging_prefix}/policy_{test_subjects}_seed{seed}")+'.pkl'
			policy_ckpt_path2 = os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{os.path.dirname(logging_prefix)}/conservative-asynchronous_single_sensor/policy_{test_subjects}_seed{seed}")+'.pkl'
			if os.path.exists(policy_ckpt_path):
				logger.info("Policy Already Trained")
				with open(policy_ckpt_path, 'rb') as file:
					policy = pickle.load(file)['best']
				logger.info(f"Policy: {policy}")
				for bp in kwargs['body_parts']:
					policy[bp] = f"conservative_{policy[bp][0]}_{policy[bp][1]}"
			# elif os.path.exists(policy_ckpt_path2):
			# 	logger.info("Policy Already Trained")
			# 	with open(policy_ckpt_path2, 'rb') as file:
			# 		policy = pickle.load(file)['best']
			# 	logger.info(f"Policy: {policy}")
			# 	for bp in kwargs['body_parts']:
			# 		policy[bp] = f"conservative_{policy[bp][0]}_{policy[bp][1]}"
			else: # otherwise, train the policy

				train_helper = PolicyTrain(active_channels, ehs, train_data_sequence, normalized_train_data_sequence,
									train_label_sequence, val_data_sequence, normalized_val_data_sequence,
									val_label_sequence,sensor_channel_map,sparse_model,**kwargs)
				
				# create a checkpoint, init sensor policies
				policy = {'current': {bp: kwargs['policy_param_init_vals'] for bp in kwargs['body_parts']},
			  			  'best_rew': {bp: [0.,0.] for bp in kwargs['body_parts']},
			  			  'best': {bp: [0.,0.] for bp in kwargs['body_parts']}}
				
				# create folder for policy checkpoint
				Path(os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{logging_prefix}")).mkdir(parents=True, exist_ok=True)
				with open(policy_ckpt_path, 'wb') as file:
					pickle.dump(policy, file)

				file_lock = multiprocessing.Lock()
				barrier = multiprocessing.Barrier(len(kwargs['body_parts']))

				processes = []
				zo_trainers = []

				# optimize sensors iteratively
				for bp in kwargs['body_parts']:
					params_bounds = [[x, y] for x, y in zip(kwargs['policy_param_min_vals'], kwargs['policy_param_max_vals'])]
					optimizer_cfg = {
						'optimizer': PatternSearch, # PatternSearch, signSGD, SGD
						'init_params': kwargs['policy_param_init_vals'],
						'lr': kwargs['policy_lr'],
						'params_bounds': params_bounds,
					}

					train_cfg = {
						'batch_size': kwargs['policy_batch_size'],
						'epochs': kwargs['policy_epochs'],
						'val_every_epochs': kwargs['policy_val_every_epochs'],
						'train_seg_duration':100*kwargs['sampling_frequency']	
					}
					
					# # load the current policy so we can get the frozen params
					# with open(ckpt_path, 'rb') as file:
					# 	frozen_policy = pickle.load(file)
					# frozen_policy.pop(bp)

					logger.info(f"Training: {bp}, policy: {policy['current']}")

					# train the policy for the given bp with others frozen
					zo_trainer = ZOTrainer(optimizer_cfg,train_cfg,train_helper,reward,logger,bp, file_lock, barrier, policy_ckpt_path)
					zo_trainers.append(zo_trainer)
					
					# zo_trainer.train()

				# bps 1->n-1 in other processes
				for i, bp in enumerate(kwargs['body_parts']):
					# skip 0, let main run 0
					if i == 0:
						continue
					p = multiprocessing.Process(target=zo_trainers[i].train,args=(i,))
					processes.append(p)
					p.start()
				
				# bp 0 in main process
				zo_trainers[0].train(0)

				for p in processes:
					p.join()
					
				# load trained policy
				with open(policy_ckpt_path, 'rb') as file:
					policy = pickle.load(file)['best']
				for bp in kwargs['body_parts']:
					policy[bp] = f"conservative_{policy[bp][0]}_{policy[bp][1]}"

			# after get policy, reset model type back to original arg
			kwargs['model_type'] = model_type
		else:
			# load classifier to use for policy evaluation
			kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
			sparse_model,_ = sparse_model_builder(**kwargs)

			# opportunistic or unconstrained for all sensors
			if kwargs['policy'] == 'unconstrained':
				stride = kwargs['unconstrained_stride']
				policy = {bp: kwargs['policy']+f"_{stride}" for bp in kwargs['body_parts']}
			elif kwargs['policy'] == 'opportunistic':
				policy = {bp: kwargs['policy'] for bp in kwargs['body_parts']}


		# ============= after getting policy, apply it to train, val, and test data
		test_packet_idxs = {}
		per_bp_data_test = {}
		for bp in kwargs['body_parts']:
			bp_channels = np.where(np.isin(active_channels,sensor_channel_map[bp]['acc']))[0]
			per_bp_data_test[bp] = normalized_test_data_sequence[:,bp_channels]
			
			test_packet_idxs[bp] = ehs.sparsify_data(policy[bp], test_data_sequence[:,bp_channels])
		
		sparse_test_ds = SparseHarDataset(per_bp_data_test, test_label_sequence, test_packet_idxs)


		# if we want to train the model, then build data loaders
		# any asynchronous_multisensor model needs to be finetuned
		if "asynchronous_multisensor" in kwargs['model_type']:
			# check if model already trained
			finetuned_model_ckpt_path = os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{logging_prefix}/finetuned_classifier_{test_subjects}_seed{seed}"+".pth")
			if os.path.exists(finetuned_model_ckpt_path):
				logger.info("Model already trained")
				kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
				sparse_model, _ = sparse_model_builder(**kwargs)
			else:
				train_packet_idxs = {}
				val_packet_idxs = {}
				per_bp_data_train = {}
				per_bp_data_val ={}
				for bp in kwargs['body_parts']:
					bp_channels = np.where(np.isin(active_channels,sensor_channel_map[bp]['acc']))[0]
					per_bp_data_train[bp] = normalized_train_data_sequence[:,bp_channels]
					per_bp_data_val[bp] = normalized_val_data_sequence[:,bp_channels]
					
					train_packet_idxs[bp] = ehs.sparsify_data(policy[bp], train_data_sequence[:,bp_channels])
					val_packet_idxs[bp] = ehs.sparsify_data(policy[bp], val_data_sequence[:,bp_channels])
				
				sparse_train_ds = SparseHarDataset(per_bp_data_train, train_label_sequence, train_packet_idxs)
				sparse_val_ds = SparseHarDataset(per_bp_data_val, val_label_sequence, val_packet_idxs)

				# create a weighted random sampler
				train_class_counts = np.bincount(sparse_train_ds.labels)
				train_class_weights = 1./train_class_counts
				train_sample_weights = np.array([train_class_weights[c] for c in sparse_train_ds.labels])

				train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=train_sample_weights, num_samples=len(train_sample_weights))

				train_loader = torch.utils.data.DataLoader(sparse_train_ds, batch_size=kwargs['finetune_batch_size'], pin_memory=False,drop_last=True,num_workers=4,sampler=train_sampler)
				val_loader = torch.utils.data.DataLoader(sparse_val_ds, batch_size=128, shuffle=False, pin_memory=False,drop_last=True,num_workers=4)

				# load the pretrained model
				# this should return one of the asynchronous_multisensor models
				kwargs['checkpoint_postfix'] = f"{test_subjects}_seed{seed}.pth"
				sparse_model, _ = sparse_model_builder(**kwargs)
				sparse_model.train()

				# train it
				finetune_args = {}
				finetune_args['epochs'] = kwargs['finetune_epochs']
				finetune_args['ese'] = finetune_args['epochs']
				finetune_args['model'] = sparse_model
				finetune_args['loss_fn'] = nn.CrossEntropyLoss()
				finetune_args['optimizer'] = torch.optim.Adam(sparse_model.parameters(),lr=kwargs['finetune_lr'])
				finetune_args['train_logname'] = f"{logging_prefix}/finetuned_classifier_{test_subjects}_seed{seed}"
				finetune_args['device'] = device
				finetune_args['train_loader'] = train_loader
				finetune_args['val_loader'] = val_loader
				finetune_args['logger'] = logger
				finetune_args['log_freq'] = 200
				finetune_args['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(finetune_args['optimizer'],kwargs['finetune_epochs'])
				train(**finetune_args)

			# load the best one
			finetuned_model_ckpt_path = os.path.join(MODEL_ROOT,"saved_data/checkpoints",f"{logging_prefix}/finetuned_classifier_{test_subjects}_seed{seed}"+".pth")
			sparse_model.load_state_dict(torch.load(finetuned_model_ckpt_path)['model_state_dict'])

		
		
		# ======================= test ==============================
		sparse_model.eval()

		active_idxs, passive_idxs = sparse_test_ds.region_decomposition()
		# print(f"Active: {len(active_idxs)/(len(active_idxs)+len(passive_idxs))}")
		# print(f"Passive: {len(passive_idxs)/(len(active_idxs)+len(passive_idxs))}")

		# next classify sparse data
		preds = np.zeros(len(test_label_sequence))
		rand_initial_pred = np.random.randint(len(np.unique(test_label_sequence)))
		last_pred = rand_initial_pred
		last_packet_idx = 0
		current_packet_idx = 0
		sent_idxs = []
		for packet_i in tqdm(range(len(sparse_test_ds))):
			packets, labels = sparse_test_ds[packet_i]
			for bp,packet in packets.items():
				if packet['age'] == 0: # most recent arrival
					at = packet['arrival_time']
					sent_idxs.append(at)
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
		logger.info(f"Passive Region: {round(passive_region,3)}, Passive Error: {round(passive_error,3)} --> {round(passive_region*passive_error,3)}\n")
		
		f1 = f1_score(test_label_sequence,preds,average=None)
		sent_idxs = np.array(sent_idxs)
		sp_l = test_label_sequence[sent_idxs]
		for i,sc in enumerate(f1):
			logger.info(f"{i}-{test_ds.selected_activity_label_map[i]}: {round(sc,3)}, frac: {round((sp_l==i).mean(),3)}")
		logger.info("==========================================\n\n")

		# print("********************* Plot ***************************")
		# plt.plot(test_label_sequence,label='labels')
		# plt.plot(preds,label='preds')
		# plt.scatter(active_idxs,np.zeros_like(active_idxs),c='g',label='active')
		# plt.scatter(passive_idxs,np.zeros_like(passive_idxs),c='r',label='passive')
		# plt.scatter(sent_idxs,np.zeros_like(sent_idxs),c='k',label='arrivals')
		# plt.show()
		results_table[subject] = (test_f1, test_acc, active_region, passive_region, active_error, passive_error)


	logger.info(f"Results: {results_table}")
	# create parent directories if needed
	kwargs['train_logname'] = os.path.join(logging_prefix,f"results_seed{seed}")
	path_items = kwargs['train_logname'].split("/")
	if  len(path_items) > 1:
		Path(os.path.join(MODEL_ROOT,"saved_data/results",*path_items[:-1])).mkdir(parents=True, exist_ok=True)

	with open(os.path.join(MODEL_ROOT,"saved_data/results",kwargs['train_logname']+".pickle"), 'wb') as file:
		pickle.dump(results_table, file)

if __name__ == '__main__':

	args = get_args()
	eval_only = args.eval
	args = vars(args)

	# organize logs by dataset and architecture
	if 'multisensor_checkpoint_prefix' in args.keys():
		args['multisensor_checkpoint_prefix'] = os.path.join(args['dataset'],args['architecture'],args['multisensor_checkpoint_prefix'])
	if 'single_sensor_checkpoint_prefix' in args.keys():
		args['single_sensor_checkpoint_prefix'] = os.path.join(args['dataset'],args['architecture'],args['single_sensor_checkpoint_prefix'])

	args['logging_prefix'] = os.path.join(args['dataset'],args['architecture'],args['logging_prefix'])

	if eval_only:
		base_path = os.path.join(MODEL_ROOT,"saved_data/results",args['logging_prefix'])
		result_logs = [os.path.join(base_path, filename) for filename in os.listdir(base_path)]
		mean, std = get_results(result_logs)
		print(f"Mean: {round(mean*100,3)}, std: {round(std*100,3)}")
	else:
		train_LOOCV(**args)
	
	