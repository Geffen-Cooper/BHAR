import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from energy_harvesting.energy_harvest import EnergyHarvester

# TODO: calculate energy required for each policy to run
class EnergyHarvestingSensor():
	""" Class to produce sparse data from HAR sensor signals

	Parameters
	----------

	eh: EnergyHarvester
		object that gives harvested energy from acceleromter traces

	packet_size: int
		the number of samples per data transmission

	leakage: float
		energy consumed per second (power) when idle
	
	sample_frequency: int
		samples per second of accelerometer data

	max_energy: float
		max energy that can be stored by sensor
	"""

	def __init__(self, 
				 eh: EnergyHarvester, 
				 packet_size: int,
				 leakage: float,
				 sample_frequency: int,
				 max_energy: float = 200e-6
				 ):
		self.eh = eh
		self.packet_size = packet_size
		self.leakage = leakage
		self.FS = sample_frequency
		self.MAX_E = max_energy

		self.dt = 1/sample_frequency
		self.LEAKAGE_PER_SAMPLE = self.dt*self.leakage

	def init_harvester(self, data: np.ndarray):
		""" Sets up harvester constants and states

		Parameters
		----------

		data: np.ndarray
			data array of dimension (L x 3) where L is the time dimension

		Returns
		-------

		None
		"""

		# create pandas data frame as specified by EnergyHarvester.power() function
		t_axis = np.arange(data.shape[0])/self.FS
		t_axis = np.expand_dims(t_axis,axis=0).T
		data = np.concatenate([t_axis,data],axis=1)
		channels = np.array([0,1,2,3]) # time + 3 acc channels of body part
		self.df = pd.DataFrame(data[:,channels],columns=['time', 'x', 'y','z'])

		# get energy as function of samples
		t_out, p_out = self.eh.power(self.df)
		e_out = self.eh.energy(t_out, p_out)
		_, self.thresh = self.eh.generate_valid_mask(e_out, self.packet_size)

		# create a mask of seen and unseen data
		self.valid = np.empty(len(e_out))
		self.valid[:] = np.nan

		# energy harvested each time step
		self.e_harvest = np.concatenate([np.array([0]),np.diff(e_out)])

		self.power_window = int(3*self.packet_size)
		cs = np.cumsum(self.e_harvest)
		window_sums = cs[self.power_window-1:] - np.concatenate(([0], cs[:-self.power_window]))
		avg_powers = window_sums / self.power_window
		self.max_p = np.max(avg_powers)

		# energy level at each time step
		self.e_trace = np.zeros(len(e_out))

		# assume a linear energy usage over the course of a packet
		# i.e., the quantity thresh/packet_size is used per sample. 
		self.linear_usage = np.linspace(0,self.thresh,self.packet_size+1)[1:]

		# provide a small margin to avoid turning off right after sending
		self.MARGIN = 5*self.LEAKAGE_PER_SAMPLE

		self.num_packets_sent = 0


	def send_packet(self, k):
		""" Alters the harvester state accordingly when want to send a packet.
		If insufficient energy is available, the time index just increments

		Parameters
		----------

		k: int
			the time index

		Returns
		-------

		k: int
			the time index

		"""
		# Check if have sufficient energy to send
		# if (self.e_trace[k] >= self.thresh + self.MARGIN + self.alpha) and (k - self.last_sent_idx >= self.tau):
		if (self.e_trace[k] >= self.thresh + self.MARGIN) and (k - self.last_sent_idx >= self.tau):
			# print(f"tau: {self.tau}, p: {self.avg_power},{self.p_n}, e: {self.e_trace[k]},{self.e_n}, params: {self.theta_power},{self.theta_energy}")
			# we are within one packet of the end of the data
			if k + self.packet_size + 1 >= len(self.e_trace):
				self.valid[k+1:] = 1
				self.e_trace[k+1:] = (-self.linear_usage[:len(self.e_trace)-k-1] + self.e_harvest[k+1:]) + self.e_trace[k]
				k = len(self.e_trace)
			else:
				# once thresh is reached, we start sampling on the next sample
				self.valid[k+1:k+1+self.packet_size] = 1

				# we apply linear energy usage for each sample and get harvested amount each step
				self.e_trace[k+1:k+1+self.packet_size] = (-self.linear_usage[:] + self.e_harvest[k+1:k+1+self.packet_size]) + self.e_trace[k]
				
				k += (self.packet_size+1)
			self.last_sent_idx = k
			self.num_packets_sent += 1
			
		else: # otherwise, move forward one time step
			k += 1

		return k

	def obtain_packets(self):
		# masking the data based on energy
		for acc in 'xyz':
			self.df[acc+'_eh'] = self.df[acc] * self.valid

		# get the transition points of the masked data to see where packets start and end
		og_data = self.df[acc+'_eh'].values
		rolled_data = np.roll(og_data, 1)
		rolled_data[0] = np.nan # in case we end halfway through a valid packet
		nan_to_num_transition_indices = np.where(~np.isnan(og_data) & np.isnan(rolled_data))[0] # arrival idxs
		num_to_nan_transition_indices = np.where(np.isnan(og_data) & ~np.isnan(rolled_data))[0] # ending idxs
		if len(num_to_nan_transition_indices) < len(nan_to_num_transition_indices):
			nan_to_num_transition_indices = nan_to_num_transition_indices[:-1]
		packet_idxs = np.stack([nan_to_num_transition_indices, num_to_nan_transition_indices]).T

		self.valid = np.nan_to_num(self.valid, nan=0)
		return packet_idxs
		

	def sparsify_data(self,
					  policy: str,
					  data: np.ndarray):
		""" Given the policy and data, generate sparsified data

		Parameters
		----------

		
		"""
		
		# initialize the energy harvesting parameters
		self.init_harvester(data)
		self.last_sent_idx = 0
		self.avg_power = 0

		# energy starts from 0 at time step 0 so simulate from timestep 1
		k = 1

		# this is a special case
		if "unconstrained" in policy:
			stride = int(policy.split("_")[1])
			start_idxs = np.arange(0,len(self.e_trace)-self.packet_size,stride)
			end_idxs = start_idxs + self.packet_size
			self.packet_idxs = np.stack([start_idxs, end_idxs]).T
			return self.packet_idxs

		if policy == "opportunistic":
			self.theta_power = 0
			self.theta_energy = 0
			
		elif "conservative" in policy:
			args = policy.split("_")
			self.theta_power, self.theta_energy = float(args[1]), float(args[2])
			# self.alpha = (50e-6)/10
			# self.tau = 100

		# -------------- start policy --------------

		# iterate over energy values
		while k < len(self.e_trace):
			# update energy state
			self.e_trace[k] = self.e_trace[k-1] + self.e_harvest[k] - self.LEAKAGE_PER_SAMPLE

			# saturate if exceed max or becomes negative
			if self.e_trace[k] > self.MAX_E:
				self.e_trace[k] = self.MAX_E
			elif self.e_trace[k] < 0: # device died
				self.e_trace[k] = 0
				self.last_sent_idx = k # reset parameter
				self.num_packets_sent = 0
				self.avg_power = 0

			if "conservative" in policy:
				self.avg_power = sum(self.e_harvest[k-self.power_window:k]) / self.power_window

				# get normalized power and energy features in [0,1]
				self.p_n = (self.avg_power - self.leakage) / (self.max_p - self.leakage)
				self.e_n = (self.e_trace[k] - self.thresh) / (self.MAX_E - self.thresh)

				self.tau = self.theta_power*self.p_n + self.theta_energy*self.e_n
			else:
				self.tau = 0

			if policy == "opportunistic" or "conservative" in policy:
				k = self.send_packet(k)

		self.packet_idxs = self.obtain_packets()
		return self.packet_idxs
		

if __name__ == '__main__':
	eh = EnergyHarvester()
	ehs = EnergyHarvestingSensor(eh,
							  	 8,
								 6.6e-6,
								 25,
								 200e-6)
	# load har datasets and then just retrieve the preprocessed raw data
	# then do gen test sequence from it
