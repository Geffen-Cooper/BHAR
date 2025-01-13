import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

class SparseHarDataset(Dataset):
    """ PyTorch Dataset class to work with sparse HAR data

	Parameters
	----------

	original_data: dict
		a dict of numpy arrays where each item is the 
        original raw data (L x C) for each body part.
        For example: {'torso' : np.ndarray(L x 3)}

	original_labels: np.ndarray
		numpy array of length L with labels for each sample

	packet_timestamps: dict
        a dict where the keys are the body parts and the
        values are N x 2 arrays with the start and end idx of each
        packet observed by the sensor
	"""

    def __init__(self,
                 original_data,
                 original_labels,
                 packet_timestamps):
        
        self.original_data = original_data
        self.original_labels = original_labels
        self.packet_timestamps = packet_timestamps
        key = list(self.packet_timestamps.keys())[0]
        self.packet_size = self.packet_timestamps[key][0][1] - self.packet_timestamps[key][0][0]

        # get the packet arrival points (end times)
        packet_arrival_times = {}
        for bp in self.packet_timestamps.keys():
            packet_arrival_times[bp] = self.packet_timestamps[bp][:,1] - 1

        # merge across body parts
        merged = []
        for bp, time_stamps in self.packet_timestamps.items():
            merged.extend([(time_stamp, bp) for time_stamp in time_stamps])
        merged.sort(key=lambda x: x[0][1])
        
        self.sorted_time_stamps = [time_stamp for time_stamp, bp in merged]
        self.packet_sources = [bp for at, bp in merged]
        
        # save the idx where each packet appears for each body part
        self.packet_source_idxs = {}
        for bp in self.original_data.keys():
            self.packet_source_idxs[bp] = np.array([idx for idx, value in enumerate(self.packet_sources) if value == bp])

        self.unique_packet_timestamps = np.unique(np.stack(self.sorted_time_stamps),axis=0)
        # self.sparse_data = []
        # # something like [{torso: (array,age), leg: (array,age)}, 
        # #                 {torso: (array,age), leg: (array,age)},
        # #                 {torso: (array,age), leg: (array,age)},...]
        # last_packet = {bp: (np.zeros((self.packet_size,3)), 0) for bp in self.original_data.keys()}
        # for time_stamp in tqdm(np.unique(np.stack(self.sorted_time_stamps),axis=0)):
        #     packet_data = {}
        #     for bp in self.original_data.keys():
        #         if time_stamp in self.packet_timestamps[bp]:
        #             st,en = time_stamp
        #             packet_data[bp] = {'data': self.original_data[bp][st:en], 
        #                                'age': 0,
        #                                'arrival_time': en}
        #         else:
        #             packet_data[bp] = {'data': last_packet[bp][0],
        #                                'age': en - last_packet[bp]['arrival_time'],
        #                                'arrival_time': en}
        #     self.sparse_data.append(packet_data)

    def region_decomposition(self):
        active_idxs = []
        passive_idxs = []

        self.sorted_arrival_times = np.stack(self.sorted_time_stamps,axis=0)[:,1]

        # get the activity transition points (include starting point idx 0)
        activity_transition_idxs = np.concatenate([np.array([0]),np.where(self.original_labels[:-1] != self.original_labels[1:])[0] + 1])

        # go through all packets
        for i in range(-1,len(self.sorted_arrival_times)-1):
            if i == -1:
                curr_packet_idx = 0 
            else:
                curr_packet_idx = self.sorted_arrival_times[i]
            next_packet_idx = self.sorted_arrival_times[i+1]

            # determine of there is a transition between the packets
            idxs_between = activity_transition_idxs[(activity_transition_idxs > curr_packet_idx) & (activity_transition_idxs < next_packet_idx)]
            if len(idxs_between) > 0: # passive region
                # everything up until the first transition is active region
                # everything after the first transition is passive region
                active_idxs.append(np.arange(curr_packet_idx,idxs_between[0]))
                passive_idxs.append(np.arange(idxs_between[0],next_packet_idx))
            else:
                # all active region
                active_idxs.append(np.arange(curr_packet_idx,next_packet_idx))
        
        # after the last packet is sent, there some remaining region left
        curr_packet_idx = next_packet_idx
        next_packet_idx = len(self.original_labels)

        # determine of there is a transition between the packets
        idxs_between = activity_transition_idxs[(activity_transition_idxs > curr_packet_idx) & (activity_transition_idxs < next_packet_idx)]
        if len(idxs_between) > 0: # passive region
            # everything up until the first transition is active region
            # everything after the first transition is passive region
            active_idxs.append(np.arange(curr_packet_idx,idxs_between[0]))
            passive_idxs.append(np.arange(idxs_between[0],next_packet_idx))
        else:
            # all active region
            active_idxs.append(np.arange(curr_packet_idx,next_packet_idx))

        self.active_region = np.concatenate(active_idxs)
        self.passive_region = np.concatenate(passive_idxs)

        return self.active_region, self.passive_region

    def __getitem__(self, idx):
        # something like [{torso: (array,age), leg: (array,age)}, 
        #                 {torso: (array,age), leg: (array,age)},
        #                 {torso: (array,age), leg: (array,age)},...]
        # start_time = time.time()
        reference_packet_timestamp = self.unique_packet_timestamps[idx]
        reference_at = reference_packet_timestamp[1]
        
        packet_data = {}
        for bp in self.original_data.keys():
            # get the nearest packet from this bp 
            ats = self.packet_timestamps[bp][:,1]
            prior_packets = (ats <= reference_at).nonzero()[0]
            
            if len(prior_packets) > 0:
                p_idx = prior_packets[-1]
                st,en = self.packet_timestamps[bp][p_idx]
                # age is how long since this packet has arrived
                age = reference_at - self.packet_timestamps[bp][p_idx][1]
                packet_data[bp] = {'data': self.original_data[bp][st:en], 
                                    'age': age,
                                    'arrival_time': self.packet_timestamps[bp][p_idx][1]}
            else:
                # there is no packet, set age to -1 to signify this
                packet_data[bp] = {'data': np.zeros((self.packet_size,3)), 
                                    'age': -1,
                                    'arrival_time': -1}
        # print(time.time()-start_time)
        label = torch.tensor(self.original_labels[reference_at]).long()

        return packet_data, label
    
    def __len__(self):
        return len(self.unique_packet_timestamps)
    

def collate_sparse_sensor_data(batch):
    print(batch)
    exit()
    packets, labels = zip(*batch)

    collated_packets = {}
    bps = packets[0].keys()
    for bp in bps:
        collated_packets[bp] = {}
        for key in batch[0][bp]:
            collated_packets[bp][key] = np.stack([item[bp][key] for item in batch], axis=0)

    labels = np.array([item[1] for item in batch])
    
    collated_packets, labels

if __name__ == '__main__':
    original_data = {f"bp{i}" : np.zeros((100,3)) for i in range(2)}
    original_labels = np.zeros(100)
    packet_timestamps = {f"bp0" : np.array([[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[30,38]]),
                         f"bp1" : np.array([[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[30,38]])+30}

    sd = SparseHarDataset(original_data,original_labels,packet_timestamps)
    
    dl = DataLoader(sd,batch_size=2)
    x = next(iter(dl))
    print(x)
    print(x[0]['bp0']['data'].shape,x[0]['bp0']['age'].shape)
    print(x[0]['bp1']['data'].shape,x[0]['bp1']['age'].shape)