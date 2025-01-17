'''
code basd on https://github.com/teco-kit/ISWC22-HAR
'''

import yaml
import os
import torch

from models.Attend import AttendDiscriminate
from models.ConvLstm import DeepConvLSTM
from utils.setup_funcs import PROJECT_ROOT, MODEL_ROOT
from models.sparse_wrappers import MultiSensorModel, SingleSensorModel, TemporalContextModel

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
    num_channels = 3*len(kwargs['body_parts'])*len(kwargs['sensors'])
    num_classes = len(kwargs['activities'])
    t_context = (len(kwargs['body_parts']),kwargs['window_size'])

    config_file = open(os.path.join(PROJECT_ROOT,'models','model_configs.yaml'), mode='r')

    if architecture == 'attend':
        config = yaml.load(config_file, Loader=yaml.FullLoader)["attend"]
        model = AttendDiscriminate(input_dim=num_channels,**config,num_class=num_classes,t_context=t_context)
        return model
    elif architecture == 'convlstm':
        config = yaml.load(config_file, Loader=yaml.FullLoader)["convlstm"]
        model = DeepConvLSTM(input_shape=(1,3,kwargs['window_size'],num_channels),nb_classes=num_classes,**config)
        return model


def sparse_model_builder(**kwargs):

    model_type = kwargs['model_type']
    device = kwargs['device']

    if kwargs['policy'] == 'conservative':
        device = 'cpu'

    if model_type == 'synchronous_multisensor':
        # this is standard HAR model
        model = model_builder(**kwargs).to(device)
        ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",kwargs['multisensor_checkpoint_prefix'],kwargs['checkpoint_postfix'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        model.eval()
        return MultiSensorModel(model,device), ckpt_path
         
    elif model_type == 'asynchronous_single_sensor':
        # this is multiple individual HAR models
        models = {}
        all_body_parts = kwargs['body_parts']
        for bp in all_body_parts:
            kwargs['body_parts'] = [bp]
            models[bp] = model_builder(**kwargs).to(device)
            ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",kwargs['single_sensor_checkpoint_prefix']+f"_{bp}",kwargs['checkpoint_postfix'])
            models[bp].load_state_dict(torch.load(ckpt_path)['model_state_dict'])
            models[bp].eval()
        return SingleSensorModel(models,device), ckpt_path
    
    elif model_type == 'asynchronous_multisensor':
        model = model_builder(**kwargs).to(device)
        ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",kwargs['multisensor_checkpoint_prefix'],kwargs['checkpoint_postfix'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        model.eval()
        return MultiSensorModel(model,device), ckpt_path

    elif model_type == 'asynchronous_multisensor_time_context':
        # this is standard HAR model
        model = model_builder(**kwargs).to(device)
        ckpt_path = os.path.join(MODEL_ROOT,f"saved_data/checkpoints/",kwargs['multisensor_checkpoint_prefix'],kwargs['checkpoint_postfix'])
        model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
        return TemporalContextModel(model,device), ckpt_path