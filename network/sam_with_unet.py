import torch
import torch.nn as nn
from network.sam_downstream_unet_3d import sam_downstream_unet_3d
from network.sam_unet_3d import sam_unet_3d

class sam_with_unet(nn.Module):
    def __init__(
        self,
        sam_channels: int = 64,
        out_channels: int = 128,
        sam_num_res_units: int = 2,
    ) -> None:
        
        super().__init__()

        self.dimensions = 3
        self.sam_channels = sam_channels
        self.out_channels = out_channels
        self.sam_num_res_units = sam_num_res_units

        self.sam = sam_unet_3d(
            num_res_units=self.sam_num_res_units,
            target_feature_size=self.sam_channels,
        )
        self.fullconn_head = sam_downstream_unet_3d(
                num_res_units=0,
                in_data_channels=1,
                in_sam_channels=self.sam_channels,
                out_channels=self.out_channels,
                disable_sam = False,
                disable_global = True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        global_feature, local_feature= self.sam(x)
        output = self.fullconn_head(x, global_feature, local_feature)

        return [global_feature, local_feature], output
    
    def load_sam_state_dict(self, state_dict):
        self.sam.load_state_dict(state_dict['net'])

    def load_fullconn_head_state_dict(self, state_dict):
        self.fullconn_head.load_state_dict(state_dict['net'])

    def freeze_sam(self):
        for param in self.sam.parameters():
            param.requires_grad = False
    
    def unfreeze_sam(self):
        for param in self.sam.parameters():
            param.requires_grad = True
    
    def get_seg_parameters(self):
        return self.fullconn_head.parameters()