# Basic dataloader for mrMonai
#
# Training images and labels should be saved in respective folders. And
# corresponding image and label files should have save name.
#
# 2023.01.04
# by Mingrui

from utility import *
import numpy as np
from data_loader.base_dataloader import base_dataloader
from monai.data import CacheDataset
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    AsDiscreted,
    Invertd,
    Activationsd,
    RandZoomd,
    RandRotated,
    CenterSpatialCropd,
    ScaleIntensityRange
)
from data_loader.flareScaleIntensityRanged import flareScaleIntensityRanged

class dataloader_seg(base_dataloader):
    '''
    Basic segmentation dataloader for mrMonai

    Training images and labels should be saved in respective folders. And \
    corresponding image and label files should have save name.
    '''

    def __init__(self, config:dict, inference=False):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
            type: (str), 'train' or 'inference'. type of dataloader.
        '''
        if not hasattr(self, 'dataloader_config'):
            self.roi_size = config['Train']['roi_size']
            if inference:
                self.dataloader_config = config['Inference']['dataloader_seg']
            else:
                self.dataloader_config = config['Train']['dataloader_seg']

        self.targets = config['Task']['seg_targets']

        if 'partial_dataset' in self.dataloader_config.keys():
            self.partial_dataset = self.dataloader_config['partial_dataset']
        else:
            self.partial_dataset = -1
        
        self.cache_rate = self.dataloader_config['cache_rate']
        self.num_workers = self.dataloader_config['num_workers']
        self.batch_size = self.dataloader_config['batch_size']
        self.num_classes = len(self.targets)
        if not inference:
            self.rand_flip = self.dataloader_config['rand_flip']
            self.rand_scale_intensity = self.dataloader_config['rand_scale_intensity']
            self.rand_shift_intensity = self.dataloader_config['rand_shift_intensity']
            self.rand_zoom = self.dataloader_config['rand_zoom']
            self.rand_rotate = self.dataloader_config['rand_rotate']

        self.pixdim = self.dataloader_config['pixdim']
        self.shuffle = True

        if self.dataloader_config['debug_model']:
            self.cache_rate = 0.0
            self.num_workers = 0
            self.shuffle = False

        super(dataloader_seg, self).__init__(config, inference=inference)

        self.init_data_list(inference)
        self.init_transforms(inference)
        
    def get_roi_size(self):
        return self.roi_size
    def get_pixdim(self):
        return self.pixdim
    
    def init_data_list(self, inference=False):
        if inference:
            if self.dataloader_config['split_method'] == 'file_assignment':
                assert 'split_filename' in self.dataloader_config, \
                    'error: dataloader_config does not have split_filename'
                split_filename = self.dataloader_config['split_filename']
                assert isfile(split_filename), 'illegal split file name.'
                split = load_json(split_filename)

                self.test_list = []
                assert 'test_group' in self.dataloader_config, \
                    'error: dataloader_config does not have test_group'
                for g in self.dataloader_config['test_group']:
                    self.test_list.extend(split[g])
                if self.partial_dataset > 0:
                    self.test_list = self.test_list[:self.partial_dataset]
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
                if self.partial_dataset > 0:
                    self.train_list = self.train_list[:self.partial_dataset]
    
    def init_transforms(self, inference=False):
        if not inference:
            # train transform
            ts_list = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="LPS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                flareScaleIntensityRanged(keys="image"),
            ]
            first_crop_size = np.array(self.roi_size)
            first_crop_size = first_crop_size * 1.3
            first_crop_size = list(np.round(first_crop_size).astype(np.int32))
            ts_list.append(RandCropByLabelClassesd(keys=["image", "label"],
                                                   label_key="label",
                                                   spatial_size=first_crop_size,
                                                   num_classes=self.num_classes))
            if self.rand_zoom:
                ts_list.append(RandZoomd(keys=["image", "label"], mode=["bilinear", "nearest"], min_zoom=0.9, max_zoom=1.1, prob=0.2))
            if self.rand_rotate:
                ts_list.append(RandRotated(keys=["image", "label"], mode=["bilinear", "nearest"], 
                                           range_x=0.3, range_y=0.3, range_z=0.3, prob=0.2))
            ts_list.append(CenterSpatialCropd(keys=["image", "label"], roi_size=self.roi_size))
            if self.rand_flip:
                ts_list.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0))
                ts_list.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1))
                ts_list.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2))
            if self.rand_scale_intensity:
                ts_list.append(RandScaleIntensityd(keys="image", factors=0.1, prob=0.4))
            if self.rand_shift_intensity:
                ts_list.append(RandShiftIntensityd(keys="image", offsets=0.1, prob=0.4))
            ts_list.append(AsDiscreted(keys="label", to_onehot=len(self.targets)+1))
            self.train_transform = Compose(ts_list)

        # val transform
        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="LPS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=self.pixdim,
                    mode=("bilinear", "nearest"),
                ),
                flareScaleIntensityRanged(keys="image"),
                AsDiscreted(keys="label", to_onehot=len(self.targets)+1),
            ]
        )
    
    def get_train_loader(self):
        train_ds = CacheDataset(
            data=self.train_list,
            transform=self.train_transform,
            # cache_num=24,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle, 
            num_workers=self.num_workers
        )

        return train_loader

    def get_val_loader(self):
        val_ds = CacheDataset(
            data=self.val_list,
            transform=self.val_transform,
            # cache_rate=self.cache_rate,
            cache_rate=0,
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
    
    def get_post_transforms(self):
        post_transforms = Compose([
            Invertd(
                keys=["pred"],
                transform=self.val_transform,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=True,
                to_tensor=True, 
                device="cpu",
            ),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", argmax=True),
        ])
        return post_transforms

if __name__ == "__main__":
    import toml
    import torch
    import numpy as np
    from monai.data import ITKWriter
    from mip_pkg.datatype.Image_Data import Image_Data
    from mip_pkg.io.IO_Image_Data import *

    config = toml.load('/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/amos_seg_2mm_test1.toml')
    dl = dataloader_seg(config)
    train_loader = dl.get_train_loader()
    output_size = 5
    for i, data in enumerate(train_loader):
        print('batch: ' + str(i))
        if i >= 5:
            break
        cache_path = '/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/debug'
        maybe_mkdir_p(cache_path)
        writer = ITKWriter(output_dtype=np.float64)
        batch_size = 1

        # landmarks = data['image_meta_dict']['landmark']

        for c in range(batch_size):
            file_path = data["image_meta_dict"]["filename_or_obj"][c]
            file_name = os.path.basename(file_path)
            # origin = data['image_meta_dict']['ori_origin'][c].numpy()
            # spacing = data['image_meta_dict']['spacing'][c].numpy()
            # input
            image_m_img = Image_Data()
            # image_m_img.Init_From_Numpy_array(data['image'][c, 0].numpy(), spacing, origin)
            image_m_img.Init_From_Numpy_array(data['image'][c, 0].numpy())
            netout_filename = join(cache_path, file_name[:-7] + "_image.nii.gz")
            Write_Image_Data(image_m_img, netout_filename)

            # image_m_img.Init_From_Numpy_array(pos_map_1.numpy(), spacing, origin)
            image_m_img.Init_From_Numpy_array(data['label'][c, 0].numpy())
            netout_filename = join(cache_path, file_name[:-7] + "_label.nii.gz")
            Write_Image_Data(image_m_img, netout_filename)
            pause = 1