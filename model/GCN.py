import torch
import torch.nn as nn

from monai.networks.layers.factories import Act, Norm
from monai.networks.nets.unet import UNet as UNet

class GCN(nn.Module):
    
    def __init__(self, unet_dict_G, unet_dict_E) -> None:
        super().__init__()

        print(unet_dict_G)
        print(unet_dict_E)

        self.model_G = UNet( 
            spatial_dims=unet_dict_G["spatial_dims"],
            in_channels=unet_dict_G["in_channels"],
            out_channels=unet_dict_G["out_channels"],
            channels=unet_dict_G["channels"],
            strides=unet_dict_G["strides"],
            num_res_units=unet_dict_G["num_res_units"],
            act=unet_dict_G["act"],
            norm=unet_dict_G["normunet"],
            dropout=unet_dict_G["dropout"],
            bias=unet_dict_G["bias"],
            )

        self.model_E = UNet(
            spatial_dims=unet_dict_E["spatial_dims"],
            in_channels=unet_dict_E["in_channels"],
            out_channels=unet_dict_E["out_channels"],
            channels=unet_dict_E["channels"],
            strides=unet_dict_E["strides"],
            num_res_units=unet_dict_E["num_res_units"],
            act=unet_dict_E["act"],
            norm=unet_dict_E["normunet"],
            dropout=unet_dict_E["dropout"],
            bias=unet_dict_E["bias"],
            )

        self.softmax = nn.Sigmoid()

    def forward(self, x):

        y_hat = self.model_G(x)
        z = torch.cat([y_hat, x], dim=1)
        y_cm = self.softmax(self.model_E(z))

        return y_hat, y_cm