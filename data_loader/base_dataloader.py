from monai.data import CacheDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from utility import *

class base_dataloader():
    '''
    the base class of all dataloader class
    '''
    def __init__(self, config:dict, inference):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        if not inference:
            self.cache_rate = self.dataloader_config['cache_rate']
            self.num_workers = self.dataloader_config['num_workers']
            self.batch_size = self.dataloader_config['batch_size']

            if 'ddp' in config['Task'] and config['Task']['ddp']:
                self.ddp = True
            else:
                self.ddp = False

        if self.dataloader_config['debug_model']:
            self.cache_rate = 0.0
            self.num_workers = 1

        self.init_data_list(inference)
        self.init_transforms(inference)

    def init_data_list(self, inference=False):
        if inference:
            if self.dataloader_config['split_method'] == 'file_assignment':
                split_filename = self.dataloader_config['split_filename']
                test_group = self.dataloader_config['test_group']
                assert isfile(split_filename), 'illegal split file name.'
                split = load_json(split_filename)
                self.test_list = []
                for g in test_group:
                    self.test_list.extend(split[g])
            else:
                assert False, 'error: unkown split_method from base_dataloaser.py'
        else:
            if self.dataloader_config['split_method'] == 'file_assignment':
                assert 'split_filename' in self.dataloader_config, \
                    'error: dataloader_config does not have split_filename'
                split_filename = self.dataloader_config['split_filename']
                assert isfile(split_filename), 'illegal split file name.'
                split = load_json(split_filename)

                self.train_list = []
                self.val_list = []
                assert 'train_group' in self.dataloader_config, \
                    'error: dataloader_config does not have train_group'
                assert 'val_group' in self.dataloader_config, \
                    'error: dataloader_config does not have val_group'
                for g in self.dataloader_config['train_group']:
                    self.train_list.extend(split[g])
                for g in self.dataloader_config['val_group']:
                    self.val_list.extend(split[g])
            else:
                assert False, 'error: unkown split_method from base_dataloaser.py'

    def get_train_loader(self):
        # here we don't cache any data in case out of memory issue
        train_ds = CacheDataset(
            data=self.train_list,
            transform=self.train_transform,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        if self.ddp:
            train_loader = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                num_workers=self.num_workers,
                sampler=DistributedSampler(train_ds),
            )
        else:
            train_loader = DataLoader(
                train_ds, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
            )

        return train_loader
    
    def get_val_loader(self):
        val_ds = CacheDataset(
            data=self.val_list,
            transform=self.val_transform,
            cache_rate=self.cache_rate,
            num_workers=1,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)
        return val_loader
    
    def get_test_loader(self):
        test_ds = CacheDataset(
            data=self.test_list,
            transform=self.val_transform,
            cache_rate=self.cache_rate,
            num_workers=1,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
        return test_loader
    