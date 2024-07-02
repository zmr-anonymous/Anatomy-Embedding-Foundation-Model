import torch
from utility import *

class base_model():
    '''
    the base class of all model class
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        self.max_epochs = int(config['Train']['max_epochs'])
        self.val_amp = config['Train']['val_amp']
        self.device = torch.device(globalVal.device)

