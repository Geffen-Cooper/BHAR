import datetime
import logging 
import time
import torch
import numpy as np
import random
import os
from pathlib import Path

def setup_paths():
    import yaml
    HOME_DIR = os.path.dirname(os.path.dirname(__file__))
    DIR_CONFIG_PATH = os.path.join(HOME_DIR, "directory_config.yaml")

    with open(DIR_CONFIG_PATH) as stream:
        paths = yaml.safe_load(stream)
    
    return paths

paths = setup_paths()

# root of project is one directory above
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
# relative path to where to save models
MODEL_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, paths['model_dir']))
# relative path to where data is saved 
# this is a dictionary with keys being dataset names
DATA_ROOT = paths['data_dir']

print("MODEL ROOT", MODEL_ROOT)

def init_logger(logname: str) -> logging.Logger:
    """ create a logger that outputs print statements to the console and a log file

    The logfile is saved to: [MODEL_ROOT]/saved_data/logs/logname_[date] [time].log

    Parameters
    ----------

    logname: str
        name for current session (this can be prefixed with a subfolder to organize
        logs for different experiments, e.g. ...logs/log_folder/logname)


    Returns
    -------

    Logger
    """
    
    # create parent directories if needed
    path_items = logname.split("/")
    if  len(path_items) > 1:
        print(MODEL_ROOT)
        Path(os.path.join(MODEL_ROOT,"saved_data/logs",*path_items[:-1])).mkdir(parents=True, exist_ok=True)

    # set the file path
    logname = logname + "_" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    fn = os.path.join(MODEL_ROOT,"saved_data/logs",str(logname)+".log")

    logger=logging.getLogger()

    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        # create the console logger
        console_handler = logging.StreamHandler()
        level = logging.INFO
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        # create the file logger
        file_handler = logging.FileHandler(fn)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        # close open file
        logger.removeHandler(logger.handlers[1])

        # create the file logger
        file_handler = logging.FileHandler(fn)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger=logging.getLogger()
    
    return logger


def init_seeds(seed: int) -> None:
    """ set a global random seed

    To make experiments reproducible this function sets all random seeds for a given session.
    Note that changing the code between two runs can affect results since functions that rely
    on random values such as dataloaders may be called in different orders and use different 
    random values. For more granualarity (e.g. fix the seed for a validation split but not for
    the training set order, refer to the specific dataloader implementations)

    Parameters
    ----------

    seed: int
        The global seed to use


    Returns
    -------

    None
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False