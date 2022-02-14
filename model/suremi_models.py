""" Full assembly of the parts to form the complete network """
from torch import nn
from .unet_parts import *
from .vit import *
from einops.layers.torch import Rearrange
import numpy as np

class suremi(nn.Module):
    def __init__(self, n_bins=64, n_conv_chan = (8,4,2,1)):
        super(suremi, self).__init__()
        self.conv33 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_bins, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            dilation=1, 
            groups=n_bins, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv33_1 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_conv_chan[0], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv55 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_bins, 
            kernel_size=5, 
            stride=1, 
            padding=2, 
            dilation=1, 
            groups=n_bins, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv55_1 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_conv_chan[1], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv77 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_bins, 
            kernel_size=7, 
            stride=1, 
            padding=3, 
            dilation=1, 
            groups=n_bins, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv77_1 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_conv_chan[2], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv99 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_bins, 
            kernel_size=9, 
            stride=1, 
            padding=4, 
            dilation=1, 
            groups=n_bins, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.conv99_1 = nn.Conv2d(
            in_channels=n_bins, 
            out_channels=n_conv_chan[3], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=1, 
            groups=1, 
            bias=True, 
            padding_mode='zeros', 
            device=None, 
            dtype=None)

        self.total_fea_chan = np.sum(n_conv_chan)
        self.w1 = nn.Conv2d(self.total_fea_chan, self.total_fea_chan, kernel_size=1)
        self.w2 = nn.Conv2d(self.total_fea_chan, self.total_fea_chan, kernel_size=1)
        self.w3 = nn.Conv2d(self.total_fea_chan, 1, kernel_size=1)
        

    def forward(self, x):
        x33 = self.conv33(x)
        x55 = self.conv55(x)
        x77 = self.conv77(x)
        x99 = self.conv99(x)

        x33_1 = self.conv33_1(x33)
        x55_1 = self.conv55_1(x55)
        x77_1 = self.conv77_1(x77)
        x99_1 = self.conv99_1(x99)

        # print(x33.size(), x55.size(), x77.size(), x99.size())
        # print(x33_1.size(), x55_1.size(), x77_1.size(), x99_1.size())
        
        x_fea = torch.cat((x33_1, x55_1, x77_1, x99_1) , dim=1, out=None).retain_grad()
        # print(x_fea.size())
        x_fea = self.w1(x_fea)
        # print(x_fea.size())
        x_fea = self.w2(x_fea)
        # print(x_fea.size())
        x_fea = self.w3(x_fea)
        # print(x_fea.size())

        return x

