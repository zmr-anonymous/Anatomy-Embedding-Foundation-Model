import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.factories import Act, Norm
from monai.utils import alias, deprecated_arg, export
from monai.networks.nets import UNet
from utility import *



class sam_unet_3d(UNet):
    def __init__(
        self,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        target_feature_size: int = 128,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:
        
        super(UNet, self).__init__()

        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = 3
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.target_feature_size = target_feature_size
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.down_layer_1 = self._get_down_layer(in_channels=1, out_channels=64, strides=2, is_top=True)
        self.down_layer_2 = self._get_down_layer(in_channels=64, out_channels=128, strides=2, is_top=False)
        self.down_layer_3 = self._get_down_layer(in_channels=128, out_channels=256, strides=2, is_top=False)
        self.down_layer_4 = self._get_down_layer(in_channels=256, out_channels=512, strides=2, is_top=False)
        self.bottom_layer = self._get_bottom_layer(in_channels=512, out_channels=512)
        out_channels_4 = max(256, self.target_feature_size)
        self.up_layer_4 = self._get_up_layer(in_channels=1024, out_channels=out_channels_4, strides=2, is_top=False)
        out_channels_3 = max(128, self.target_feature_size)
        self.up_layer_3 = self._get_up_layer(in_channels=256+out_channels_4, out_channels=out_channels_3, strides=2, is_top=False)
        out_channels_2 = max(64, self.target_feature_size)
        self.up_layer_2 = self._get_up_layer(in_channels=128+out_channels_3, out_channels=out_channels_2, strides=2, is_top=False)
        self.up_layer_1 = self._get_up_layer(in_channels=64+out_channels_2, out_channels=self.target_feature_size, strides=2, is_top=True)

        self.global_full_connection = nn.Conv3d(out_channels_4, self.target_feature_size,kernel_size=1,padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_1 = self.down_layer_1(x)
        down_2 = self.down_layer_2(down_1)
        down_3 = self.down_layer_3(down_2)
        down_4 = self.down_layer_4(down_3)
        bottom = self.bottom_layer(down_4)
        up_4 = self.up_layer_4(torch.cat([bottom, down_4], dim=1))
        up_3 = self.up_layer_3(torch.cat([up_4, down_3], dim=1))
        up_2 = self.up_layer_2(torch.cat([up_3, down_2], dim=1))
        up_1 = self.up_layer_1(torch.cat([up_2, down_1], dim=1))

        local_out = F.normalize(up_1)
        global_out = F.normalize(self.global_full_connection(up_4))
        
        return [global_out, local_out]
