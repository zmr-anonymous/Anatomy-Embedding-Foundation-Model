import torch
from utility import *

class base_inference():
    '''
    the base class of all inference class
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''

        # data loader
        dataloader_name = config['Inference']['data_loader']
        dataloader_class = recursive_find_class(['data_loader'], dataloader_name, 'data_loader')
        assert dataloader_class is not None, "error: data loader class not found. (from inference.py)"
        self.dataloader = dataloader_class(config, True)

        # model
        model_name = config['Train']['model']
        model_class = recursive_find_class(['model'], model_name, 'model')
        assert model_class is not None, "error: model class not found. (from inference.py)"
        self.model = model_class(config)

        self.network = self.model.get_network()
        self.model_name = config['Inference']['model_name']
        self.device = torch.device(globalVal.device)
        # https://blog.csdn.net/weixin_43019673/article/details/122350210
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.val_amp = config['Inference']['val_amp']
        