import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from experiments.zero_order_algos import signSGD, SGD, PatternSearch
from utils.setup_funcs import PROJECT_ROOT
import pickle

torch.set_printoptions(sci_mode=True)

class ZOTrainer():
	def __init__(self, optimizer_cfg, train_cfg, policy_trainer, reward_function, logger, frozen_sensor_params, training_bp):
		self.optimizer_cfg = optimizer_cfg
		self.train_cfg = train_cfg

		self.policy_trainer = policy_trainer
		self.reward_function = reward_function
		self.logger = logger
		self.frozen_sensor_params = frozen_sensor_params

		self.bp = training_bp

		self._build_optimizer()
		self.logger.info("\n=========== Policy Training ===========")
	
	def _build_optimizer(self):
		# Initialize optimizer
		Optimizer = self.optimizer_cfg['optimizer']
		self.optimizer = Optimizer(self.optimizer_cfg['init_params'], self.optimizer_cfg['lr'], self.train_cfg['batch_size'], self.reward_function, params_bounds=self.optimizer_cfg['params_bounds'])

	def optimize_model(self, *f_args):
		return self.optimizer.forward(*f_args)

	def train_one_epoch(self, iteration, writer):

		# sample train segment
		data, data_n, labels = self.policy_trainer.sample_train_segment(self.train_cfg['train_seg_duration'])
		
		f_args = {
			'frozen_sensor_params': self.frozen_sensor_params,
			'per_bp_data': data,
			'per_body_part_data_normalized': data_n,
			'label_sequence': labels,
			'harvesting_sensor': self.policy_trainer.harvesting_sensor,
			'reward_type': 'active_region',
			'classifier': self.policy_trainer.model,
			'train_mode':True
		} 
		
		average_reward = self.optimize_model(f_args)
		
		self.logger.info("Iteration {}: avg reward: {:.3f}, params: {}, epsilon: {}".format(iteration, average_reward, self.optimizer.params, self.optimizer.epsilon))

		writer.add_scalar("train_metric/average_reward", average_reward, iteration)
		# need to generalize this for multiple body parts
		# writer.add_scalars("train_metric/params", 
		#                   {'alpha': self.optimizer.params[0],
		#                    'tau': self.optimizer.params[1]}, iteration)
		return average_reward

	def train(self):
		writer = SummaryWriter(self.policy_trainer.runs_path)
		original_epsilon = self.optimizer.epsilon
		best_params = self.optimizer.params

		self.policy_trainer.model.eval()

		self.logger.info(f"Original Epsilon: {original_epsilon}")

		val_loss = self.validate(0,writer)
		best_val_reward = val_loss['avg_reward']

		for iteration in tqdm(range(self.train_cfg['epochs'])):
			self.train_one_epoch(iteration, writer)
			if iteration % self.train_cfg['val_every_epochs'] == 0 and iteration > 0:
				val_loss = self.validate(iteration, writer)
				if val_loss['avg_reward'] >= best_val_reward:
					self.logger.info(f"Saving new best parameters {self.optimizer.params}, reward: {val_loss['avg_reward']} > {best_val_reward} (f1: {val_loss['f1']})")
					best_params = self.optimizer.params
					best_val_reward = val_loss['avg_reward']
					# save params: load checkpoint, update, save checkpoint
					with open(self.policy_trainer.checkpoint_path+'.pkl', 'rb') as file:
						policy = pickle.load(file)
					policy[self.bp] = best_params
					with open(self.policy_trainer.checkpoint_path+'.pkl', 'wb') as file:
						pickle.dump(policy, file)
			
			if (self.optimizer.epsilon <= original_epsilon * 1e-2).all():
				self.logger.info(f"Stopping training as reached convergence with epsilon {self.optimizer.epsilon}")
				self.logger.info(f"Parameters are {best_params}")
				break

		self.logger.info(f"best reward: {best_val_reward}")
		return best_params
			
	def validate(self, iteration, writer):

		f_args = {
			'frozen_sensor_params': self.frozen_sensor_params,
			'per_bp_data': self.policy_trainer.per_bp_data_val,
			'per_body_part_data_normalized': self.policy_trainer.per_bp_data_val_n,
			'label_sequence': self.policy_trainer.val_labels,
			'harvesting_sensor': self.policy_trainer.harvesting_sensor,
			'reward_type': 'accuracy',
			'classifier': self.policy_trainer.model,
			'train_mode':False
		} 

		reward, f1 = self.reward_function(self.optimizer.params,**f_args)

		self.logger.info("Iteration {}: params: {}, val_policy_f1: {:.3f}".format(iteration, self.optimizer.params, f1))

		if writer is not None:
			writer.add_scalar("val_metric/policy_f1", f1, iteration)
			writer.add_scalar("val_metric/policy_reward", reward, iteration)

		val_loss = {
			'f1': f1,
			'avg_reward': reward,
		}

		
		# policy_sample_times = (learned_packets[0]).long()
		# opp_sample_times = (opp_packets[0]).long()
		# self.fig.suptitle(r"$\alpha = {:.3e}, \tau = {:.3e}$".format(self.optimizer.params[0], self.optimizer.params[1]))
		# self.axs.plot(t_axis, learned_e_trace)
		# self.axs.plot(t_axis, opp_e_trace, linestyle='--')
		# self.axs.scatter(t_axis[policy_sample_times], learned_e_trace[policy_sample_times], s=100, label='policy')
		# self.axs.scatter(t_axis[opp_sample_times], opp_e_trace[opp_sample_times], marker='D', s=100, alpha=0.3, label='opp')
		# self.axs.axhline(y=self.sensor.thresh, linestyle='--', color='green') # Opportunistic policy will send at this energy
		# self.axs.set_xlabel("Time")
		# self.axs.set_ylabel("Energy")
		# self.axs.legend()
		# plt.tight_layout()
		# plt.savefig(f"{self.plot_dir}/plot_{iteration}.png")
		# self.axs.cla()

		return val_loss