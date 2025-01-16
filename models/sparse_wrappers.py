'''
This file contains wrapper modules for HAR models
to handle sparse data input during the forward pass
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F

class MultiSensorModel(nn.Module):
    """ This is a wrapper architecture for standard HAR models.
        It just reformats the data during the forward pass by concatenating
        the most recent packet from each sensor

		Parameters
		----------
		
		multisensor_model: nn.Module
			the model to wrap

	"""
    def __init__(self, multisensor_model):
        super(MultiSensorModel,self).__init__()

        self.multisensor_model = multisensor_model

    def forward(self,x):
        # fmerge the data across body parts
        # ** we assume the order of the body parts here
        # is the same as the order of the channels used when
        # training the model
        packet_data = []
        for bp,packet in x.items():
            packet_data.append(packet['data'])
        
        # merge channels, then convert to float tensor with batch dimension
        packet_data = torch.tensor(np.concatenate(packet_data,axis=1)).float()

        if len(packet_data.shape) == 2: # no batch dimension
            packet_data = packet_data.unsqueeze(0)

        return self.multisensor_model(packet_data)


class SingleSensorModel(nn.Module):
    """ This is a wrapper architecture to combine multiple
        single sensor models and only do inference using the one
        corresponding to the most recent packet

		Parameters
		----------
		
		single_sensor_models: dict
			a dict of the single sensor models where
            the body part is the key

	"""
    def __init__(self, single_sensor_models):
        super(SingleSensorModel,self).__init__()

        self.single_sensor_models = single_sensor_models

    def forward(self,x):
        # find which body part got the latest packet
        # and forward pass through that model
        for bp,packet in x.items():
            if packet['age'] == 0: # most recent arrival
                # we only do inference with this model so need add batch dimension (no data loader)
                packet_data = torch.tensor(packet['data']).float().unsqueeze(0)
                return self.single_sensor_models[bp](packet_data)


class TemporalContextModel(nn.Module):
    """ This is a wrapper architecture for standard HAR models
        which receive temporally asynchronous packets at inference.
        It also concatenates the packets like MultiSensorModel but
        has a second input, the age of each packet, during the forward
        pass.

		Parameters
		----------
		
		multisensor_model: nn.Module
			the model to wrap

	"""
    def __init__(self, multisensor_model):
        super(TemporalContextModel,self).__init__()

        self.multisensor_model = multisensor_model

    def forward(self,x):
        # merge the data across body parts
        # ** we assume the order of the body parts here
        # is the same as the order of the channels used when
        # training the model. Also get the ages of each packet
        packet_data = []
        ages = []
        for bp,packet in x.items():
            packet_data.append(packet['data'])
            ages.append(packet['age'])
        
        # merge channels, then convert to float tensor
        packet_data = torch.tensor(np.concatenate(packet_data,axis=1)).float()
        age_data = torch.tensor(ages).float()
        if len(packet_data.shape) == 2: # no batch dimension
            packet_data = packet_data.unsqueeze(0)
            age_data = age_data.unsqueeze(0)
        
        # then do forward
        return self.multisensor_model(packet_data, age_data)