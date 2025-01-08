'''
code basd on https://github.com/teco-kit/ISWC22-HAR
'''

import yaml
import os
import torch

from models.Attend import AttendDiscriminate
from models.ConvLstm import DeepConvLSTM
from utils.setup_funcs import PROJECT_ROOT
from models.sparse_wrappers import DenseModel, MultiSensor

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
    elif architecture == 'convlstm':
        config = yaml.load(config_file, Loader=yaml.FullLoader)["convlstm"]
        model = DeepConvLSTM(input_shape=(1,3,kwargs['window_size'],num_channels),nb_classes=num_classes,**config)
        return model


def sparse_model_builder(**kwargs):

    model_type = kwargs['model_type']

    if model_type == 'dense_synchronous_baseline':
        # this is standard HAR model
        model = model_builder(**kwargs)
        ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/",kwargs['checkpoint_prefix'],kwargs['checkpoint_postfix'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        return DenseModel(model)
         
    elif model_type == 'sparse_asychronous_baseline':
        # this is multiple individual HAR models
        models = {}
        all_body_parts = kwargs['body_parts']
        for bp in all_body_parts:
            kwargs['body_parts'] = [bp]
            models[bp] = model_builder(**kwargs)
            ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/",kwargs['checkpoint_prefix']+f"_{bp}",kwargs['checkpoint_postfix'])
            models[bp].load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        return MultiSensor(models)
    elif model_type == 'sparse_asychronous_contextualized':
        pass