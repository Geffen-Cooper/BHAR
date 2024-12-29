import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from pathlib import Path

from models.model_builder import model_builder
from datasets.dataset import HARClassifierDataset, load_har_classifier_dataloaders
from datasets.preprocess_raw_data import preprocess_DSADS, preprocess_RWHAR, preprocess_PAMAP2
from experiments.train import train, validate
from utils.setup_funcs import PROJECT_ROOT, init_logger, init_seeds

def get_args():
	parser = argparse.ArgumentParser(
			description="Dataset and model arguments for training HAR classifiers",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)

	parser.add_argument("--logging_prefix", default="logfile", type=str, help="name for this training session")
	parser.add_argument(
			"--architecture",
			default="attend",
			type=str,
			choices=["attend", "tinyhar", "convlstm","simple"],
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
	parser.add_argument("--window_size", default=8, type=int, help="sliding window size")
	parser.add_argument("--overlap_frac", default=0.5, type=float, help="fraction of window to overlap")
	parser.add_argument("--batch_size", default=128, type=int, help="training batch size")
	parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
	parser.add_argument("--epochs", default=50, type=int, help="training epochs")
	parser.add_argument("--ese", default=10, type=int, help="early stopping epochs")
	parser.add_argument("--log_freq", default=200, type=int, help="after how many batches to log")


	args = parser.parse_args()

	return args


def train_LOOCV(**kwargs):
	""" Trains N models for Leave One Subject Out Cross Validation

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

		train_loader,val_loader,test_loader = load_har_classifier_dataloaders(train_subjects, test_subjects, **kwargs)
	
		logger.info(f"Train Samples: {len(train_loader.dataset)}")
		logger.info(f"Val Samples: {len(val_loader.dataset)}")
		logger.info(f"Test Samples: {len(test_loader.dataset)}")

		# init training parameters
		model = model_builder(**kwargs)
		kwargs['model'] = model
		kwargs['loss_fn'] = nn.CrossEntropyLoss()
		kwargs['optimizer'] = torch.optim.AdamW(model.parameters(),lr=kwargs['lr'])
		kwargs['train_logname'] = f"{logging_prefix}/{train_subjects}_seed{seed}"
		kwargs['device'] = device
		kwargs['train_loader'] = train_loader
		kwargs['val_loader'] = val_loader
		kwargs['logger'] = logger
		kwargs['lr_scheduler'] = torch.optim.lr_scheduler.CosineAnnealingLR(kwargs['optimizer'],kwargs['epochs'])

		# load the model if already trained
		ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/",f"{kwargs['train_logname']}.pth")
		if os.path.exists(ckpt_path):
			model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		else:
			# otherwise train the model
			train(**kwargs)
		
		# load the one with the best validation accuracy and evaluate on test set
		model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		test_acc,test_f1,test_loss = validate(model, test_loader, device, kwargs['loss_fn'])
		logger.info(f"Test F1: {test_f1}, Test Acc: {test_acc}")
		logger.info("==========================================\n\n")
		results_table[subject] = (test_f1, test_acc)


	logger.info(f"Results: {results_table}")
	# create parent directories if needed
	path_items = kwargs['train_logname'].split("/")
	if  len(path_items) > 1:
		Path(os.path.join(PROJECT_ROOT,"saved_data/results",*path_items[:-1])).mkdir(parents=True, exist_ok=True)

	with open(os.path.join(PROJECT_ROOT,"saved_data/results",kwargs['train_logname'],"results.pickle"), 'wb') as file:
		pickle.dump(results_table, file)

if __name__ == '__main__':

	args = get_args()
	args = vars(args)

	# organize logs by dataset and architecture
	args['logging_prefix'] = os.path.join(args['dataset'],args['architecture'],args['logging_prefix'])

	train_LOOCV(**args)
	
	