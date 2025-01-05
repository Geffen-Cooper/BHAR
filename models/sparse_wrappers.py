'''
This file contains wrapper modules for HAR models
to handle sparse data input during the forward pass
'''

import numpy as np
import torch
import torch.nn as nn

class DenseModel(nn.Module):
    """ This is a wrapper architecture for standard HAR models.
        It just reformats the data during the forward pass

		Parameters
		----------
		
		dense_model: nn.Module
			the model to wrap

	"""
    def __init__(self, dense_model):
        super(DenseModel,self).__init__()

        self.dense_model = dense_model

    def forward(self,x):
        # fmerge the data across body parts
        # ** we assume the order of the body parts here
        # is the same as the order of the channels used when
        # training the model
        packet_data = []
        for bp,packet in x.items():
            packet_data.append(packet['data'])
        
        # merge channels, then convert to float tensor with batch dimension
        packet_data = torch.tensor(np.concatenate(packet_data,axis=1)).float().unsqueeze(0)
        return self.dense_model(packet_data)


class MultiSensor(nn.Module):
    """ This is a wrapper architecture to combine multiple
        single sensor models

		Parameters
		----------
		
		sensor_models: dict
			a dict of the single sensor models where
            the body part is the key

	"""
    def __init__(self, sensor_models):
        super(MultiSensor,self).__init__()

        self.sensor_models = sensor_models

    def forward(self,x):
        # find which body part got the latest packet
        # and forward pass through that model
        for bp,packet in x.items():
            if packet['age'] == 0: # most recent arrival
                packet_data = torch.tensor(packet['data']).float().unsqueeze(0)
                return self.sensor_models[bp](packet_data)
