import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.layers.factories import Act, Norm
from monai.utils import alias, deprecated_arg, export
from monai.networks.nets import UNet
from utility import *



class sam_downstream_unet_3d(UNet):
    def __init__(
        self,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        in_data_channels: int = 1,
        in_sam_channels: int = 64,
        out_channels: int = 128,
        disable_sam:bool = False,
        disable_global:bool = False,
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
        self.in_data_channels = in_data_channels
        self.in_sam_channels = in_sam_channels
        self.out_channels = out_channels
        self.disable_sam = disable_sam
        self.disable_global = disable_global
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        if not self.disable_sam:
            self.down_layer_data_1 = self._get_down_layer(in_channels=self.in_data_channels, out_channels=32, strides=2, is_top=True)
            self.down_layer_sam_1 = self._get_down_layer(in_channels=self.in_sam_channels, out_channels=32, strides=2, is_top=True)
            self.down_layer_2 = self._get_down_layer(in_channels=32+32, out_channels=64, strides=2, is_top=False)
            # self.down_layer_2 = self._get_down_layer(in_channels=32, out_channels=64, strides=2, is_top=False)
        else:
            self.down_layer_data_1 = self._get_down_layer(in_channels=self.in_data_channels, out_channels=32, strides=2, is_top=True)
            self.down_layer_2 = self._get_down_layer(in_channels=32, out_channels=64, strides=2, is_top=False)

        self.down_layer_3 = self._get_down_layer(in_channels=64, out_channels=128, strides=2, is_top=False)
        if not (self.disable_sam or self.disable_global):
            self.down_layer_4 = self._get_down_layer(in_channels=192, out_channels=256, strides=2, is_top=False)
        else:
            self.down_layer_4 = self._get_down_layer(in_channels=128, out_channels=256, strides=2, is_top=False)
        self.bottom_layer = self._get_bottom_layer(in_channels=256, out_channels=512)
        self.up_layer_4 = self._get_up_layer(in_channels=768, out_channels=128, strides=2, is_top=False)
        self.up_layer_3 = self._get_up_layer(in_channels=256, out_channels=64, strides=2, is_top=False)
        self.up_layer_2 = self._get_up_layer(in_channels=128, out_channels=32, strides=2, is_top=False)

        if not self.disable_sam:
            self.up_layer_1 = self._get_up_layer(in_channels=32+64, out_channels=self.out_channels, strides=2, is_top=True)
            # self.up_layer_1 = self._get_up_layer(in_channels=32+32, out_channels=self.out_channels, strides=2, is_top=True)
        else:
            self.up_layer_1 = self._get_up_layer(in_channels=64, out_channels=self.out_channels, strides=2, is_top=True)

    def forward(self, data: torch.Tensor, sam_global: torch.Tensor, sam_local: torch.Tensor) -> torch.Tensor:

        if not self.disable_sam:
            down_data_1 = self.down_layer_data_1(data)
            down_sam_1 = self.down_layer_sam_1(sam_local)
            down_1 = torch.cat([down_data_1, down_sam_1], dim=1)
            # down_1 = down_sam_1
        else:
            down_data_1 = self.down_layer_data_1(data)
            down_1 = down_data_1

        down_2 = self.down_layer_2(down_1)
        down_3 = self.down_layer_3(down_2)
        if not (self.disable_sam or self.disable_global):
            down_4 = self.down_layer_4(torch.cat([down_3, sam_global], dim=1))
        else:
            down_4 = self.down_layer_4(down_3)
        bottom = self.bottom_layer(down_4)
        up_4 = self.up_layer_4(torch.cat([bottom, down_4], dim=1))
        up_3 = self.up_layer_3(torch.cat([up_4, down_3], dim=1))
        up_2 = self.up_layer_2(torch.cat([up_3, down_2], dim=1))
        up_1 = self.up_layer_1(torch.cat([up_2, down_1], dim=1))

        # out = F.normalize(up_1)
        
        return up_1
