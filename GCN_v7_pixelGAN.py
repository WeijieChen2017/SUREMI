
# Generative Confidential Network

import os
import gc
import copy
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import requests

from monai.networks.nets.unet import UNet as UNet
from monai.networks.layers.factories import Act, Norm
import bnn

from utils import add_noise, weighted_L1Loss
from model import GCN

class Unet_sigmoid(nn.Module):
    
    def __init__(self, unet_dict_E) -> None:
        super().__init__()

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
        return self.softmax(self.model_E(x))


model_list = [
    ["GCN_v7_pixelGAN_abs", [6], 0., ],
    ]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)
# current_model_idx = 0
# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = model_list[current_model_idx][0]
train_dict["gpu_ids"] = model_list[current_model_idx][1]
train_dict["dropout"] = model_list[current_model_idx][2]

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [96, 96, 96]
train_dict["epochs"] = 200
train_dict["batch"] = 16
train_dict["well_trained_model"] = "./project_dir/Unet_Monai_Iman_v2/model_best_181.pth"

train_dict["beta"] = 1e6 # resize KL loss
train_dict["model_term"] = "Monai_Unet3d"
train_dict["dataset_ratio"] = 1
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False


unet_dict_E = {}
unet_dict_E["spatial_dims"] = 3
unet_dict_E["in_channels"] = 2
unet_dict_E["out_channels"] = 1
unet_dict_E["channels"] = (32, 64, 128, 256)
unet_dict_E["strides"] = (2, 2, 2)
unet_dict_E["num_res_units"] = 4
unet_dict_E["act"] = Act.PRELU
unet_dict_E["normunet"] = Norm.INSTANCE
unet_dict_E["dropout"] = 0.0
unet_dict_E["bias"] = True
train_dict["model_E"] = unet_dict_E


train_dict["folder_X"] = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"
train_dict["folder_Y"] = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"
# train_dict["pre_train"] = "swin_base_patch244_window1677_kinetics400_22k.pth"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "BCELoss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_E = Unet_sigmoid(unet_dict_E)

# model_E = UNet( 
#     spatial_dims=unet_dict_E["spatial_dims"],
#     in_channels=unet_dict_E["in_channels"],
#     out_channels=unet_dict_E["out_channels"],
#     channels=unet_dict_E["channels"],
#     strides=unet_dict_E["strides"],
#     num_res_units=unet_dict_E["num_res_units"],
#     act=unet_dict_E["act"],
#     norm=unet_dict_E["normunet"],
#     dropout=unet_dict_E["dropout"],
#     bias=unet_dict_E["bias"],
#     )

model_E.train()
model_E = model_E.to(device)

# optim = torch.optim.RMSprop(model_E.parameters(), lr=train_dict["opt_lr"])
bin_loss = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.AdamW(
    model_E.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

train_list = glob.glob(train_dict["folder_X"]+"*_xtr.nii.gz")
val_list = glob.glob(train_dict["folder_X"]+"*_xva.nii.gz")
test_list = glob.glob(train_dict["folder_X"]+"*_xte.nii.gz")

data_division_dict = {
    "train_list_X" : train_list,
    "val_list_X" : val_list,
    "test_list_X" : test_list}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0
ONE_CM = torch.ones((
    train_dict["batch"],
    1, 
    train_dict["input_size"][0],
    train_dict["input_size"][1],
    train_dict["input_size"][2])).float().to(device)
# wandb.watch(model)

package_train = [train_list, True, False, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]:

        file_list = package[0]
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model_E.train()
        else:
            model_E.eval()

        random.shuffle(file_list)
        
        case_loss = np.zeros((len(file_list)))

        # N, C, D, H, W
        x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            y_path = file_path.replace("x", "y")
            z_path = file_path.replace("x", "z")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, cnt_file+1, len(file_list)), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            z_file = nib.load(z_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()
            z_data = z_file.get_fdata()

            batch_x = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
            batch_y = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
            batch_z = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

            for idx_batch in range(train_dict["batch"]):
                
                d0_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][1])
                d1_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][2])
                d2_offset = np.random.randint(x_data.shape[2] - train_dict["input_size"][0])

                x_slice = x_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                y_slice = y_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                z_slice = z_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                batch_x[idx_batch, 0, :, :, :] = x_slice
                batch_y[idx_batch, 0, :, :, :] = y_slice
                batch_z[idx_batch, 0, :, :, :] = z_slice


            fusion_map = np.random.uniform(0, 1, size=batch_y.shape)
            fusion_map[fusion_map<0.5] = 0.
            fusion_map[fusion_map>0.5] = 1.
            fusion = np.zeros(batch_y.shape)
            fusion[fusion_map == 0.] = batch_z[fusion_map == 0.]
            fusion[fusion_map == 1.] = batch_y[fusion_map == 1.]
            
            batch_xf = np.concatenate([batch_x, fusion], axis=1)
            batch_fmap = torch.from_numpy(fusion_map).float().to(device)
            batch_xf = torch.from_numpy(batch_xf).float().to(device)

            # batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)
            batch_z = torch.from_numpy(batch_z).float().to(device)

            
            if isTrain:

                optim.zero_grad()
                fmap_hat = model_E(batch_xf)
                loss = bin_loss(fmap_hat, batch_fmap, weight=torch.abs(batch_y-batch_z))
                loss.backward()
                optim.step()
                case_loss[cnt_file] = loss.item()
                print("Loss: ", case_loss[cnt_file])

            if isVal:

                with torch.no_grad():
                    fmap_hat = model_E(batch_xf)
                    loss = bin_loss(fmap_hat, batch_fmap)

                case_loss[cnt_file] = loss.item()
                print("Loss: ", case_loss[cnt_file])

        epoch_loss = np.mean(case_loss[cnt_file])
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("Loss: ", epoch_loss)
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_xf.npy", batch_xf.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_fmap.npy", batch_fmap.cpu().detach().numpy())
            torch.save(model_E, train_dict["save_folder"]+"model_curr.pth".format(idx_epoch + 1))
            
            if epoch_loss < best_val_loss:
                # save the best model
                torch.save(model_E, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optim, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = epoch_loss

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()
