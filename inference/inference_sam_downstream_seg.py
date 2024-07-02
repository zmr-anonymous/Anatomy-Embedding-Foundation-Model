import time
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from inference.inference_seg import inference_seg
from network.sam_with_unet import sam_with_unet
from mip_pkg.datatype.Image_Data import Image_Data
import mip_pkg.io.IO_Image_Data as IO_Image_Data
from utility import *

class inference_sam_downstream_seg(inference_seg):
    '''
    general inference
    '''
    def __init__(self, config:dict):
        '''
        constructor function

        Args:
            config: (dict), config of the whole task
        '''
        super(inference_sam_downstream_seg, self).__init__(config)
        self.disabel_sam = self.model.get_disable_sam()

        if not self.disabel_sam:
            self.sam_target_feature_size = config['Train']['trainer_SAM_downstream_seg']['sam_target_feature_size']
            self.sam_num_res_units = config['Train']['trainer_SAM_downstream_seg']['sam_num_res_units']
            self.checkpoint_name = config['Train']['trainer_SAM_downstream_seg']['sam_model_name']
            self.sam_network_name = config['Train']['trainer_SAM_downstream_seg']['network_name']

            assert isfile(self.checkpoint_name), "error: checkpoint file does not exist. from trainer_SAM_downstream_seg.py"
            checkpoint = torch.load(self.checkpoint_name)
            
            if self.sam_network_name == 'sam_with_unet':
                self.sam_network = sam_with_unet(
                    sam_channels=self.sam_target_feature_size,
                    out_channels=14,
                ).to(globalVal.device)

            if self.sam_network_name == 'sam_unet_3d' or self.sam_network_name == 'sam_unet_3d_2':
                self.sam_network.load_state_dict(checkpoint['net'])
            else:
                filtered_dict = {key.replace('sam.', ''): value for key, value in checkpoint['net'].items() if 'sam.' in key}
                self.sam_network.sam.load_state_dict(filtered_dict)
                self.sam_network.eval()
        

    def inference_sliding_window(self, input):
        '''
        method to inference data with current model
        '''
        def _foreword(input):
            if self.disabel_sam:
                outputs = self.network(input, None, None)
            else:
                if self.sam_network_name == 'sam_unet_3d':
                    global_feature, local_feature = self.sam_network(input)
                    outputs = self.network(input, global_feature, local_feature)
                elif self.sam_network_name == 'sam_unet_3d_2':
                    local_feature, _ = self.sam_network(input)
                    outputs = self.network(input, _, local_feature)
                elif self.sam_network_name == 'sam_with_fullconn_head':
                    feature, _ = self.sam_network(input)
                    global_feature, local_feature = feature
                    outputs = self.network(input, global_feature, local_feature)
                elif self.sam_network_name == 'sam_with_unet' or self.sam_network_name == 'sam_with_cluster':
                    feature, _ = self.sam_network(input)
                    global_feature, local_feature = feature
                    outputs = self.network(input, global_feature, local_feature)
            
            return outputs

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=self.dataloader.get_roi_size(),
                sw_batch_size=1,
                predictor=_foreword,
                overlap=0.5,
                # device=torch.device('cpu')
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
                        # spacing = self.dataloader.get_pixdim()

                        image_data = Image_Data()
                        image_data.Init_From_Numpy_array(val_data_post['pred'][0].cpu().numpy(), origin=origin,spacing=spacing)
                        file_full_name = join(globalVal.output_path, file_name+'.nii.gz')
                        IO_Image_Data.Write_Image_Data(image_data, file_full_name)

                # progress bar
                print("\r", end="")
                s = time()-data_start
                print(f"inference progress: {i}/" + num_of_data + f" time: {s:.4f} s", end="")
                sys.stdout.flush()

        