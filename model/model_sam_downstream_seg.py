import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from utility import *

from model.base_model import base_model
from monai.networks.nets import UNet
from network.sam_downstream_unet_3d import sam_downstream_unet_3d

class model_sam_downstream_seg(base_model):
    '''
    
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        if not hasattr(self, 'model_config'):
            self.model_config = config['Train']['model_sam_downstream_seg']
            self.out_channels = len(config['Task']['seg_targets'])+1
            self.num_res_units = self.model_config['num_res_units']
            self.in_sam_channels = self.model_config['in_sam_channels']
            self.disable_sam = self.model_config['disable_sam']
            self.disable_global = self.model_config['disable_global']
            self.network_name = self.model_config['network_name']
        super(model_sam_downstream_seg, self).__init__(config)

        if self.network_name == 'sam_downstream_unet_3d':
            self.network = sam_downstream_unet_3d(
                num_res_units=self.num_res_units,
                in_data_channels=1,
                in_sam_channels=self.in_sam_channels,
                out_channels=self.out_channels,
                disable_sam = self.disable_sam,
                disable_global = self.disable_global,
            ).to(self.device)
        else:
            assert False, 'unknown network'

        self.optimizer = torch.optim.Adam(self.network.parameters(), 1e-4, weight_decay=1e-5)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)
    
    def get_target_feature_size(self):
        return self.target_feature_size
    
    def get_network(self):
        return self.network

    def get_optimizer(self):
        return self.optimizer

    def get_lr_scheduler(self):
        return self.lr_scheduler

    def get_disable_sam(self):
        return self.disable_sam
