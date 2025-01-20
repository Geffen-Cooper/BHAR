import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from experiments.zero_order_algos import signSGD, SGD, PatternSearch
from utils.setup_funcs import PROJECT_ROOT
import pickle

torch.set_printoptions(sci_mode=True)

class ZOTrainer():
	def __init__(self, optimizer_cfg, train_cfg, policy_trainer, reward_function, logger, training_bp, file_lock, barrier, ckpt_path):
		self.optimizer_cfg = optimizer_cfg
		self.train_cfg = train_cfg

		self.policy_trainer = policy_trainer
		self.reward_function = reward_function
		self.logger = logger
		self.frozen_sensor_params = {}

		self.file_lock = file_lock
		self.barrier = barrier

		self.bp = training_bp

		self.ckpt_path = ckpt_path

		# all other body parts are frozen parameters
		with file_lock:
			# save params: load checkpoint, update, save checkpoint
			with open(self.ckpt_path, 'rb') as file:
				policy = pickle.load(file)['current']
				for bp in policy.keys():
					if bp != self.bp:
						self.frozen_sensor_params[bp] = policy[bp]
		

		self._build_optimizer()
		self.logger.info(f"\nBP: {self.bp} =========== Policy Training ===========")
	
	def _build_optimizer(self):
		# Initialize optimizer
		Optimizer = self.optimizer_cfg['optimizer']
		self.optimizer = Optimizer(self.optimizer_cfg['init_params'], self.optimizer_cfg['lr'], self.train_cfg['batch_size'], self.reward_function, params_bounds=self.optimizer_cfg['params_bounds'])

	def optimize_model(self, *f_args):
		return self.optimizer.forward(*f_args)

	def train_one_epoch(self, iteration, writer):

		# sample train segment
		data, data_n, labels = self.policy_trainer.sample_train_segment(self.train_cfg['train_seg_duration'])
		# bps = list(data.keys())
		# plt.plot(data[bps[0]])
		# plt.savefig("test.png")
		
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
		
		self.logger.info("BP: {}, Iteration {}: avg reward: {:.3f}, params: {}, epsilon: {}".format(self.bp, iteration, average_reward, self.optimizer.params, self.optimizer.epsilon))

		writer.add_scalar("train_metric/average_reward", average_reward, iteration)
		return average_reward

	def write_val(self,val):
		with self.file_lock:
			with open(self.ckpt_path, 'rb') as file:
				policy = pickle.load(file)
				policy['current'][self.bp] = (val,self.optimizer.params)
			with open(self.ckpt_path, 'wb') as file:
				pickle.dump(policy, file)
	
	def update_best_rew(self):
		with self.file_lock:
			with open(self.ckpt_path, 'rb') as file:
				policy = pickle.load(file)
			# which update was the best
			vals = np.array([policy['current'][bp][0] for bp in policy['current'].keys()])
			bps = list(policy['current'].keys())
			best_bp = bps[np.argmax(vals)]
			# update only these params
			policy['best_rew'][best_bp] = policy['current'][best_bp][1]
			with open(self.ckpt_path, 'wb') as file:
				pickle.dump(policy, file)
			self.logger.info(f"Step - best bp: {best_bp}, policy: {policy['best_rew']}")
	
	def update_policy(self):
		with self.file_lock:
			with open(self.ckpt_path, 'rb') as file:
				policy = pickle.load(file)
			# set current and frozen params
			# current params is either set to the best or reset to what it was on the last iteration
			# which is what is saved in 'best_rew'
			if isinstance(policy['best_rew'][self.bp], torch.Tensor):
				self.optimizer.params = policy['best_rew'][self.bp]
			else:
				self.optimizer.params = torch.tensor(policy['best_rew'][self.bp])
			# set all the frozen params to old/updated values
			for bp in policy['current'].keys():
				if bp != self.bp:
					self.frozen_sensor_params[bp] = policy['best_rew'][bp]
			self.logger.info(f"synch: {self.optimizer.params},{self.frozen_sensor_params}")


	def train(self,p_id):
		writer = SummaryWriter(self.policy_trainer.runs_path)
		original_epsilon = self.optimizer.epsilon
		best_params = self.optimizer.params

		self.policy_trainer.model.eval()

		self.logger.info(f"BP: {self.bp}, Original Epsilon: {original_epsilon}")
		self.logger.info("Opportunistic")
		# only one process needs to validate from init
		if p_id == 0:
			val_loss = self.validate(0,writer)
			best_val_reward = val_loss['avg_reward']
			best_val_f1 = val_loss['f1']
		# wait for validation to finish
		self.barrier.wait()

		for iteration in tqdm(range(self.train_cfg['epochs'])):

			# let each sensor take a step
			avg_rew = self.train_one_epoch(iteration, writer)

			# save the current parameters and avg reward on this step
			self.write_val(avg_rew)

			# wait until all threads updated parameters to synchronize
			# before reading the updated parameters
			self.barrier.wait()

			# only one process needs to determine the best update
			if p_id == 0:
				self.update_best_rew()
			self.barrier.wait()

			# let all sensors (processes) update their policy
			self.update_policy()
			self.barrier.wait()
			

			if iteration % self.train_cfg['val_every_epochs'] == 0 and iteration > 0:
				if p_id == 0:
					val_loss = self.validate(iteration, writer)
					if val_loss['f1'] > best_val_f1:
						# self.logger.info(f"BP: {self.bp}, Saving new best parameters {self.optimizer.params}, reward: {val_loss['avg_reward']} > {best_val_reward} (f1: {val_loss['f1']})")
						# best_params = self.optimizer.params

						with self.file_lock:
							# save params: load checkpoint, update, save checkpoint
							with open(self.ckpt_path, 'rb') as file:
								policy = pickle.load(file)
							policy['best'] = policy['best_rew']
							with open(self.ckpt_path, 'wb') as file:
								pickle.dump(policy, file)
							self.logger.info(f"Saving new best parameters {policy['best']}, f1: {val_loss['f1']} > {best_val_f1} (reward: {val_loss['avg_reward']})")

						best_val_f1 = val_loss['f1']
					else:
						with self.file_lock:
							with open(self.ckpt_path, 'rb') as file:
								policy = pickle.load(file)
							self.logger.info(f"f1: {val_loss['f1']} < {best_val_f1}, {policy['best']}")
			# wait for validation to finish
			self.barrier.wait()
			
			if (self.optimizer.epsilon <= original_epsilon * 1e-2).all():
				self.logger.info(f"Stopping training as reached convergence with epsilon {self.optimizer.epsilon}")
				self.logger.info(f"Parameters are {best_params}")
				break

		# self.logger.info(f"BP: {self.bp}, best reward: {best_val_reward}")
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

		self.logger.info("Iteration: {}, val_policy_f1: {:.3f}\n".format(iteration, f1))

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