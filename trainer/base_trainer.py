import torch
from utility import *
import loss.loss_helper

class base_trainer():
    '''
    the base class of all trainer class
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        self.config = config
        # train data loader

        dataloader_name = config['Train']['data_loader']
        dataloader_class = recursive_find_class(['data_loader'], dataloader_name, 'data_loader')
        assert dataloader_class is not None, "error: data loader class not found. (from run_training.py)"
        self.dataloader = dataloader_class(config)
        
        # model
        self.ddp_mode = 'ddp' in config['Task'] and config['Task']['ddp']
        self.generate_model()
        if self.ddp_mode:
            self.ddp_init()

        # loss
        self.loss_function = loss.loss_helper.get_loss(config)

        # https://blog.csdn.net/weixin_43019673/article/details/122350210
        torch.multiprocessing.set_sharing_strategy('file_system')
    
    def ddp_init(self):
        self.network = torch.nn.parallel.DistributedDataParallel(
            self.network, 
            device_ids=[self.config['Task']['local_rank']], 
            output_device=self.config['Task']['local_rank'])

    def generate_model(self):
        model_name = self.config['Train']['model']
        model_class = recursive_find_class(['model'], model_name, 'model')
        assert model_class is not None, "error: model class not found. (from run_training.py)"
        self.model = model_class(self.config)
        self.network = self.model.get_network()
        self.optimizer = self.model.get_optimizer()
        self.lr_scheduler = self.model.get_lr_scheduler()

    def get_network(self):
        return self.network
        
        