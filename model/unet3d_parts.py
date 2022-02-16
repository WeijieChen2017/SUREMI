""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUp(nn.Module):
    """ DoubleConv -> Up"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.bilinear = bilinear
        self.double_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=(1,2,2), stride=(1,2,2))

    def forward(self, x):
        return self.up(self.double_conv(x))


class UpConv(nn.Module):
    """ Up -> DoubleConv """

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        self.bilinear = bilinear

        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)

        self.double_conv = DoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        
    def forward(self, x):
        return self.double_conv(self.up(x))



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels

#         self.double_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.double_conv3 = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.double_conv5 = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#         self.output = nn.Sequential(
#             nn.Conv2d(out_channels*3, out_channels, kernel_size=5, padding=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):

#         x1 = self.double_conv1(x)
#         x3 = self.double_conv3(x)
#         x5 = self.double_conv5(x)

#         return self.output(torch.cat((x1, x3, x5), 1))


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        # x = self.sigmoid(x)
        x = self.conv1(x)
        return x
