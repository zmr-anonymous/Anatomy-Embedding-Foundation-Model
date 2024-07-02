# Basic dataloader for mrMonai
#
# Training images and labels should be saved in respective folders. And
# corresponding image and label files should have save name.
#
# 2023.01.04
# by Mingrui

from utility import *
import numpy as np
from data_loader.dataloader_seg import dataloader_seg
from data_loader.LoadPreprocessed import LoadPreprocessed
from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandZoomd,
    RandRotated,
    AsDiscreted,
    RandRotate90d,
)

class dataloader_seg_preprocessed(dataloader_seg):
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
                self.dataloader_config = config['Inference']['dataloader_seg_preprocessed']
            else:
                self.dataloader_config = config['Train']['dataloader_seg_preprocessed']
        super(dataloader_seg_preprocessed, self).__init__(config, inference=inference)

    def init_transforms(self, inference=False):
        if not inference:
            # train transform
            ts_list = []
            ts_list.append(LoadPreprocessed(keys=["image", "label"], mmap_mode=False))
            first_crop_size = np.array(self.roi_size)
            first_crop_size = first_crop_size * 1.5
            first_crop_size = list(np.round(first_crop_size).astype(np.int32))
            ts_list.append(RandCropByLabelClassesd(keys=["image", "label"],
                                                   label_key="label",
                                                   spatial_size=first_crop_size,
                                                   num_classes=self.num_classes))
            if self.rand_zoom:
                ts_list.append(RandZoomd(keys=["image", "label"], mode=["trilinear", "nearest"], min_zoom=0.9, max_zoom=1.1, prob=0.2))
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
                LoadPreprocessed(keys=["image", "label"]),
                AsDiscreted(keys="label", to_onehot=len(self.targets)+1),
            ]
        )    

if __name__ == "__main__":
    import toml
    import torch
    import numpy as np
    from monai.data import ITKWriter
    from mip_pkg.datatype.Image_Data import Image_Data
    from mip_pkg.io.IO_Image_Data import *

    config = toml.load('/home/mingrui/my_nas/Monai/202404_AnatomyFormer/config_files/Flare_nnFormer_200.toml')
    config['Train']['dataloader_seg_preprocessed']['debug_model'] = True
    dl = dataloader_seg_preprocessed(config)
    train_loader = dl.get_train_loader()
    output_size = 5
    for i, data in enumerate(train_loader):
        print('batch: ' + str(i))
        # if i >= 5:
        #     break
        # cache_path = '/home/mingrui/disk1/projects/monai/Projects/202402_MICCAI/debug'
        # maybe_mkdir_p(cache_path)
        # writer = ITKWriter(output_dtype=np.float64)
        # batch_size = 1

        # # landmarks = data['image_meta_dict']['landmark']

        # for c in range(batch_size):
        #     file_path = data["image_meta_dict"]["filename_or_obj"][c]
        #     file_name = os.path.basename(file_path)
        #     # origin = data['image_meta_dict']['ori_origin'][c].numpy()
        #     # spacing = data['image_meta_dict']['spacing'][c].numpy()
        #     # input
        #     image_m_img = Image_Data()
        #     # image_m_img.Init_From_Numpy_array(data['image'][c, 0].numpy(), spacing, origin)
        #     image_m_img.Init_From_Numpy_array(data['image'][c, 0].numpy())
        #     netout_filename = join(cache_path, file_name[:-7] + "_image.nii.gz")
        #     Write_Image_Data(image_m_img, netout_filename)

        #     # image_m_img.Init_From_Numpy_array(pos_map_1.numpy(), spacing, origin)
        #     image_m_img.Init_From_Numpy_array(data['label'][c, 0].numpy())
        #     netout_filename = join(cache_path, file_name[:-7] + "_label.nii.gz")
        #     Write_Image_Data(image_m_img, netout_filename)
        #     pause = 1