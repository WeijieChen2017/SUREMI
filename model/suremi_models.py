""" Full assembly of the parts to form the complete network """
from torch import nn
from .unet_parts import *
from .vit import *
from einops.layers.torch import Rearrange

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
        

    def forward(self, x):
        x33 = self.conv33(x)
        x55 = self.conv55(x)
        x77 = self.conv77(x)
        x99 = self.conv99(x)

        print(x33.size(), x55.size(), x77.size(), x99.size())
        return x

