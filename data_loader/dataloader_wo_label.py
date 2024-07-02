# Basic dataloader for mrMonai
#
# Training images and labels should be saved in respective folders. And
# corresponding image and label files should have save name.
#
# 2023.01.04
# by Mingrui

import torch
import pickle
import numpy as np
from typing import Dict, Hashable, Mapping, Union, Sequence, Optional
from data_loader.dataloader_seg import dataloader_seg
from monai.transforms.transform import MapTransform, Randomizable
from torch.utils.data.distributed import DistributedSampler
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    AsDiscreted,
    DivisiblePadd,
    MapTransform,
    ScaleIntensityRange,
    InvertibleTransform,
)
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from utility import *
from data_loader.flareScaleIntensityRanged import flareScaleIntensityRanged

class ReadAndRandSpatialCropd(MapTransform, InvertibleTransform, Randomizable):
    """
    
    """

    def __init__(
        self,
        keys: KeysCollection,
        roi_size: Union[Sequence[int], int],
        do_not_crop:bool = False,        
    ) -> None:
        self.roi_size = roi_size        
        self.keys = keys
        self.do_not_crop = do_not_crop

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        # the first key must exist to execute random operations
        shape = np.load(d[self.first_key(d)], mmap_mode='r').shape[1:]
        basename = os.path.basename(d[self.first_key(d)])
        basename = basename.split('.')[0]
        with open(join(os.path.dirname(d[self.first_key(d)]), basename+'.pkl'), 'rb') as f:
            d['image_meta_dict'] = pickle.load(f)

        rand = np.random.rand(3)
        if not self.do_not_crop:
            s = np.floor((np.array(shape) - np.array(self.roi_size))*rand).astype(np.int32)

        for key in self.keys:
            d[key] = np.load(d[key], mmap_mode='r')
            if not self.do_not_crop:
                d[key] = d[key][:, s[0]:s[0]+self.roi_size[0], s[1]:s[1]+self.roi_size[1], s[2]:s[2]+self.roi_size[2]]
                d[key] = torch.from_numpy(d[key].copy())
            else:
                d[key] = d[key].copy()
        return d

# class amosScaleIntensityRanged(MapTransform):

#     backend = ScaleIntensityRange.backend

#     def __init__(
#         self,
#         keys: KeysCollection,
#         dtype: DtypeLike = np.float32,
#         allow_missing_keys: bool = False,
#     ) -> None:
#         super().__init__(keys, allow_missing_keys)
#         self.scaler_1 = ScaleIntensityRange(a_min=990, a_max=1140, b_min=-1, b_max=1, clip=True, dtype=dtype)
#         self.scaler_2 = ScaleIntensityRange(a_min=-500, a_max=750, b_min=-1, b_max=1, clip=True, dtype=dtype)
#         # self.scaler_2 = ScaleIntensityRange(a_min=-100, a_max=350, b_min=-1, b_max=1, clip=True, dtype=dtype)

#     def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
#         d = dict(data)
#         for key in self.key_iterator(d):
#             if torch.min(d[key]) > -10:
#                 d[key] = self.scaler_1(d[key])
#             else:
#                 d[key] = self.scaler_2(d[key])
#         return d

class dataloader_wo_label(dataloader_seg):
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
                self.dataloader_config = config['Inference']['dataloader_wo_label']
            else:
                self.dataloader_config = config['Train']['dataloader_wo_label']
        if not inference:
            if 'ddp' in config['Task'] and config['Task']['ddp']:
                self.ddp = True
            else:
                self.ddp = False

        self.preprocessed = self.dataloader_config['preprocessed']
        super(dataloader_wo_label, self).__init__(config, inference)

        if self.preprocessed:
            self.cache_rate = 0.0
        
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

    def init_transforms(self, inference=False):
        if not inference:
            # train transform
            if not self.preprocessed:
                ts_list = [
                    LoadImaged(keys=["image"]),
                    EnsureChannelFirstd(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="LPS"),
                    Spacingd(
                        keys=["image"],
                        pixdim=self.pixdim,
                        mode=("bilinear"),
                    ),
                    flareScaleIntensityRanged(keys="image"),
                    RandSpatialCropd(keys=["image"], roi_size=self.roi_size, random_size=False),
                    DivisiblePadd(keys=["image"], k=self.roi_size, mode='constant', constant_values=0)
                ]
            else:
                ts_list = [
                    ReadAndRandSpatialCropd(keys=["image"], roi_size=self.roi_size),
                    DivisiblePadd(keys=["image"], k=self.roi_size, mode='constant', constant_values=0)
                ]
            if self.rand_flip:
                ts_list.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=0))
                ts_list.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=1))
                ts_list.append(RandFlipd(keys=["image"], prob=0.5, spatial_axis=2))
            # ts_list.append(NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)),
            if self.rand_scale_intensity:
                ts_list.append(RandScaleIntensityd(keys="image", factors=0.1, prob=1.0))
                ts_list.append(RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0))
            self.train_transform = Compose(ts_list)

        # val transform
        if not self.preprocessed:
            self.val_transform = Compose(
                [
                    LoadImaged(keys=["image"],image_only=False),
                    EnsureChannelFirstd(keys=["image"]),
                    EnsureTyped(keys=["image"]),
                    Orientationd(keys=["image"], axcodes="LPS"),
                    Spacingd(
                        keys=["image",],
                        pixdim=self.pixdim,
                        mode=("bilinear"),
                    ),
                    flareScaleIntensityRanged(keys="image"),
                    # ts_list.append(NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)),
                    # AsDiscreted(keys="label", to_onehot=len(self.targets)+1),
                ]
            )
        else:
            self.val_transform = Compose(
                [
                    ReadAndRandSpatialCropd(keys=["image"], roi_size=None, do_not_crop=True),
                    # AsDiscreted(keys="label", to_onehot=len(self.targets)+1),
                ]
            )

    def init_data_list(self, inference=False):
        super(dataloader_wo_label, self).init_data_list(inference)
        if hasattr(self, 'train_list'):
            for i in range(len(self.train_list)):
                if 'label' in self.train_list[i].keys():
                    del self.train_list[i]['label']
        if hasattr(self, 'val_list'):
            for i in range(len(self.val_list)):
                if 'label' in self.val_list[i].keys():
                    del self.val_list[i]['label']

        # self.test_list = self.test_list[98:]

    def get_train_loader(self):
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
                num_workers=self.num_workers
            )

        return train_loader


if __name__ == "__main__":
    import toml
    import torch
    import numpy as np
    from monai.data import ITKWriter
    from mip_pkg.datatype.Image_Data import Image_Data
    from mip_pkg.io.IO_Image_Data import *

    config = toml.load('/home/mingrui/disk1/projects/monai/Projects/202302_FedSAM/config_files/amos_ct_seg_2mm_sam_20_7.toml')
    dl = dataloader_seg_amos(config)
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
            netout_filename = join(cache_path, file_name[:-4] + "_image.nii.gz")
            Write_Image_Data(image_m_img, netout_filename)

            # image_m_img.Init_From_Numpy_array(pos_map_1.numpy(), spacing, origin)
            image_m_img.Init_From_Numpy_array(data['label'][c, 0].numpy())
            netout_filename = join(cache_path, file_name[:-4] + "_label.nii.gz")
            Write_Image_Data(image_m_img, netout_filename)
            pause = 1