'''
code basd on https://github.com/teco-kit/ISWC22-HAR
'''

import yaml
import os

from models.Attend import AttendDiscriminate
from utils.setup_funcs import PROJECT_ROOT

def model_builder(**kwargs):
    """ Initializes the specified architecture

		Parameters
		----------
		
		**kwargs:
			has dataset and training specific parameters (e.g. number of classes and input channels)

		Returns
		-------

        model: nn.Module
            the initialized model
	"""

    architecture = kwargs['architecture']
    num_channels = num_channels = 3*len(kwargs['body_parts'])*len(kwargs['sensors'])
    num_classes = len(kwargs['activities'])

    config_file = open(os.path.join(PROJECT_ROOT,'models','model_configs.yaml'), mode='r')

    if architecture == 'attend':
        config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]
        model = AttendDiscriminate(input_dim=num_channels,**config,num_class=num_classes)
        return model
