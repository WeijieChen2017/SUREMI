# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

import random

__all__ = ["UNet_shuffle"]


@export("monai.networks.nets")
@alias("Unet")
class UNet_shuffle(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """

    @deprecated_arg(
        name="dimensions", new_name="spatial_dims", since="0.6", msg_suffix="Please use `spatial_dims` instead."
    )
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        is_WDO: bool = False,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_WDO = is_WDO


        # UNet( 
        # spatial_dims=unet_dict["spatial_dims"],
        # in_channels=unet_dict["in_channels"],
        # out_channels=unet_dict["out_channels"],
        # channels=unet_dict["channels"],
        # strides=unet_dict["strides"],
        # num_res_units=unet_dict["num_res_units"],
        # act=unet_dict["act"],
        # norm=unet_dict["normunet"],
        # dropout=unet_dict["dropout"],
        # bias=unet_dict["bias"],
        # )

        # input - down1 ------------- up1 -- output
        #         |                   |
        #         down2 ------------- up2
        #         |                   |
        #         down3 ------------- up3
        #         |                   |
        #         down4 -- bottom --  up4
        # 1 -> (32, 64, 128, 256) -> 1

        self.down1 = ResidualUnit(3, self.in_channels, self.channels[0], self.strides[0],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down2 = ResidualUnit(3, self.channels[0], self.channels[1], self.strides[1],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.down3 = ResidualUnit(3, self.channels[1], self.channels[2], self.strides[2],
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.bottom = ResidualUnit(3, self.channels[2], self.channels[3], 1,
                kernel_size=self.kernel_size, subunits=self.num_res_units,
                act=self.act, norm=self.norm, dropout=self.dropout,
                bias=self.bias, adn_ordering=self.adn_ordering)
        self.up3 = nn.Sequential(
                Convolution(3, self.channels[3]+self.channels[2], self.channels[1], strides=self.strides[2],
                kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, 
                bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
                ResidualUnit(3, self.channels[1], self.channels[1], strides=1,
                kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm,
                dropout=self.dropout, bias=self.bias, last_conv_only=False, adn_ordering=self.adn_ordering,))
        self.up2 = nn.Sequential(
                Convolution(3, self.channels[1]*2, self.channels[0], strides=self.strides[1],
                kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, 
                bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
                ResidualUnit(3, self.channels[0], self.channels[0], strides=1,
                kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm,
                dropout=self.dropout, bias=self.bias, last_conv_only=False, adn_ordering=self.adn_ordering,))
        self.up1 = nn.Sequential(
                Convolution(3, self.channels[0]*2, self.out_channels, strides=self.strides[0],
                kernel_size=self.up_kernel_size, act=self.act, norm=self.norm, dropout=self.dropout, 
                bias=self.bias, conv_only=False, is_transposed=True, adn_ordering=self.adn_ordering),
                ResidualUnit(3, self.out_channels, self.out_channels, strides=1,
                kernel_size=self.kernel_size, subunits=1, act=self.act, norm=self.norm,
                dropout=self.dropout, bias=self.bias, last_conv_only=True, adn_ordering=self.adn_ordering))

        # self.down1 = nn.ModuleList(self.down1)
        # self.down2 = nn.ModuleList(self.down2)
        # self.down3 = nn.ModuleList(self.down3)
        # self.bottom = nn.ModuleList(self.bottom)
        # self.up3 = nn.ModuleList(self.up3)
        # self.up2 = nn.ModuleList(self.up2)
        # self.up1 = nn.ModuleList(self.up1)
        
    def channel_dropout(self, x, is_WDO = False):

        len_channel = x.size()[1]
        if not is_WDO:
            idx = torch.randperm(x.shape[1])
            x = x[:, idx, :, :, :]
        else:
            score = torch.mean(x, axis=(0, 2, 3, 4))
            score = torch.divide(1, score)
            base = torch.rand(score.size()).to(torch.device('cuda'))
            score = torch.pow(base, score)
            order = torch.argsort(score)
            x = x[:, order, :, :, :]

        return x

    def forward(self, x: torch.Tensor, order:Sequence[int] = []) -> torch.Tensor:

        # print(x.size())
        x1 = self.channel_dropout(self.down1(x), self.is_WDO)
        # print(x1.size())
        x2 = self.channel_dropout(self.down2(x1), self.is_WDO)
        # print(x2.size())
        x3 = self.channel_dropout(self.down3(x2), self.is_WDO)
        # print(x3.size())
        xb = self.channel_dropout(self.bottom(x3), self.is_WDO)
        # print(xb.size())
        x5 = self.channel_dropout(self.up3(torch.cat([x3, xb], dim=1)), self.is_WDO)
        # print(x5.size())
        x6 = self.channel_dropout(self.up2(torch.cat([x2, x5], dim=1)), self.is_WDO)
        # print(x6.size())
        x7 = self.up1(torch.cat([x1, x6], dim=1))
        # print(x7.size())

        # print(x.size())         torch.Size([4, 1, 96, 96, 96])
        # print(x1.size())        torch.Size([4, 32, 48, 48, 48])
        # print(x2.size())        torch.Size([4, 64, 24, 24, 24])
        # print(x3.size())        torch.Size([4, 128, 12, 12, 12])
        # print(xb.size())        torch.Size([4, 256, 12, 12, 12])
        # print(x5.size())        torch.Size([4, 64, 24, 24, 24])
        # print(x6.size())        torch.Size([4, 32, 48, 48, 48])
        # print(x7.size())        torch.Size([4, 1, 96, 96, 96])

        return x7
