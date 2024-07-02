from torch import nn
from utility import *

class base_loss(nn.Module):
    '''
    the base class of all loss class
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        pass
        
