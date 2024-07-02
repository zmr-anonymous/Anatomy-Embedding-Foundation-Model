import time
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine
from monai.data import ITKWriter, decollate_batch, ImageWriter
from inference.base_inference import base_inference
from utility import *
from mip_pkg.datatype.Image_Data import Image_Data
import mip_pkg.io.IO_Image_Data as IO_Image_Data

class inference_seg(base_inference):
    '''
    general inference
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        super(inference_seg, self).__init__(config)

        self.data_loader = self.dataloader.get_test_loader()
        self.post_transforms = self.dataloader.get_post_transforms()
        if 'targets' in config['Task'].keys():
            self.targets = config['Task']['targets']
        self.save_label_file = config['Inference']['save_label_file']
        self.sliding_window_device = config['Inference']['sliding_window_device']
        if self.sliding_window_device == 'None':
            self.sliding_window_device = None
        

    def inference_sliding_window(self, input):
        '''
        method to inference data with current model
        '''
        def _foreword(input):
            outputs = self.network(input)
            return outputs

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.dataloader.get_roi_size(),
                sw_batch_size=1,
                predictor=_foreword,
                overlap=0.5,
                device=self.sliding_window_device,
            )
        if self.val_amp:
            with torch.cuda.amp.autocast():
                val_outputs = _compute(input)
        else:
            val_outputs = _compute(input)

        return val_outputs
    
    def inference(self):
        checkpoint = torch.load(os.path.join(globalVal.model_path, self.model_name))
        self.network.load_state_dict(checkpoint['net'])
        self.network.eval()

        with torch.no_grad():
            num_of_data = str(len(self.data_loader))
            for i, val_data in enumerate(self.data_loader):
                data_start = time()
                val_inputs = val_data["image"].to(self.device)
                val_data["pred"] = self.inference_sliding_window(val_inputs)
                val_data = decollate_batch(val_data)
                if self.save_label_file:
                    for index, one_val_outputs in enumerate(val_data):
                        val_data_post = self.post_transforms(one_val_outputs)
                        oir_file_name = val_data[index]["image_meta_dict"]["filename_or_obj"]
                        file_name = os.path.basename(oir_file_name)
                        file_name = file_name.split(".")[0]
                        ori_image_data = IO_Image_Data.Read_Image_Data(oir_file_name)
                        origin = np.array(ori_image_data.Get_Origin())
                        spacing = np.array(ori_image_data.Get_Spacing())

                        image_data = Image_Data()
                        image_data.Init_From_Numpy_array(val_data_post['pred'][0].cpu().numpy(), origin=origin,spacing=spacing)
                        file_full_name = join(globalVal.output_path, file_name+'.nii.gz')
                        IO_Image_Data.Write_Image_Data(image_data, file_full_name)

                # progress bar
                print("\r", end="")
                s = time()-data_start
                print(f"inference progress: {i}/" + num_of_data + f" time: {s:.4f} s", end="")
                sys.stdout.flush()

        