import os
import gc
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

import torch
import torchvision
import requests

from model import SwinTransformer3D

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "Swin3d_Iman"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["channel"] = 18
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["gpu_ids"] = [0]
train_dict["epochs"] = 100
train_dict["batch"] = 1
train_dict["dropout"] = 0
train_dict["model_term"] = "SwinTransformer3D"
train_dict["deconv_channels"] = 6

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["pre_train"] = "model_best_282.pth"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"pred/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Swin-B
# model = SwinTransformer3D(
#     pretrained=None,
#     pretrained2d=True,
#     patch_size=(2,4,4),
#     in_chans=3,
#     embed_dim=128,
#     depths=[2, 2, 18, 2],
#     num_heads=[4, 8, 16, 32],
#     window_size=(16,7,7),
#     mlp_ratio=4.,
#     qkv_bias=True,
#     qk_scale=None,
#     drop_rate=0.,
#     attn_drop_rate=0.,
#     drop_path_rate=0.2,
#     norm_layer=nn.LayerNorm,
#     patch_norm=True,
#     frozen_stages=-1,
#     use_checkpoint=False,
#     deconv_channels = 6)

model = torch.load(train_dict["save_folder"]+train_dict["pre_train"], map_location=torch.device('cpu'))

model.train()
model = model.to(device)
criterion = nn.SmoothL1Loss()

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr = train_dict["opt_lr"],
#     betas = train_dict["opt_betas"],
#     eps = train_dict["opt_eps"],
#     weight_decay = train_dict["opt_weight_decay"],
#     amsgrad = train_dict["amsgrad"]
#     )

# ==================== data division ====================

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

selected_list = np.asarray(X_list)
np.random.shuffle(selected_list)
selected_list = list(selected_list)

val_list = selected_list[:int(len(selected_list)*train_dict["val_ratio"])]
val_list.sort()
test_list = selected_list[-int(len(selected_list)*train_dict["test_ratio"]):]
test_list.sort()
train_list = list(set(selected_list) - set(val_list) - set(test_list))
train_list.sort()

data_division_dict = {
    "train_list_X" : train_list,
    "val_list_X" : val_list,
    "test_list_X" : test_list}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e6
# wandb.watch(model)

# print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))
package_test = [test_list, False, False, "test"]

for package in [package_test]:

    file_list = package[0]
    isTrain = package[1]
    isVal = package[2]
    iter_tag = package[3]

    model.eval()

    # random.shuffle(file_list)

    # n c d h w

    for cnt_file, file_path in enumerate(file_list):

        x_path = file_path
        file_name = os.path.basename(file_path)
        x_file = nib.load(x_path)
        x_data = x_file.get_fdata()
        len_z = x_data.shape[2]
        idx_z = 0

        pred = np.zeros(x_data.shape)

        batch_x = np.zeros((1, 3, train_dict["channel"], x_data.shape[0], x_data.shape[1]))
        cnt_channel = 0

        while idx_z < len_z:

            # print("idx_z:", idx_z, "cnt_channe;", cnt_channel)
            if idx_z == 0:
                batch_x[:, 0, cnt_channel, :, :] = x_data[:, :, 0]
                batch_x[:, 1, cnt_channel, :, :] = x_data[:, :, 0]
                batch_x[:, 2, cnt_channel, :, :] = x_data[:, :, 1]
            elif idx_z == len_z - 1:
                batch_x[:, 0, cnt_channel, :, :] = x_data[:, :, len_z - 2]
                batch_x[:, 1, cnt_channel, :, :] = x_data[:, :, len_z - 1]
                batch_x[:, 2, cnt_channel, :, :] = x_data[:, :, len_z - 1]
            else:
                batch_x[:, 0, cnt_channel, :, :] = x_data[:, :, idx_z-1]
                batch_x[:, 1, cnt_channel, :, :] = x_data[:, :, idx_z]
                batch_x[:, 2, cnt_channel, :, :] = x_data[:, :, idx_z+1]

            idx_z += 1
            cnt_channel += 1
            if cnt_channel == train_dict["channel"]:
                # slices fill a full batch
                batch_x = torch.from_numpy(batch_x).float().to(device)
                y_hat = model(batch_x).cpu().detach().numpy()
                batch_x = np.zeros((1, 3, train_dict["channel"], x_data.shape[0], x_data.shape[1]))
                for idx_rz in range(train_dict["channel"]):
                    # print("idx_rz", idx_z-idx_rz-1, train_dict["channel"]-idx_rz-1)
                    pred[:, :, idx_z-idx_rz-1] = np.squeeze(y_hat[:, 1, train_dict["channel"]-idx_rz-1, :, :])
                cnt_channel = 0

        if cnt_channel > 0:
            batch_x = torch.from_numpy(batch_x).float().to(device)
            y_hat = model(batch_x).cpu().detach().numpy()
            for idx_rz in range(cnt_channel):
                # print("idx_rz", idx_rz)
                pred[:, :, idx_z-idx_rz-1] = np.squeeze(y_hat[:, 1, cnt_channel-idx_rz-1, :, :])
            
        pred_file = nib.Nifti1Image(pred, x_file.affine, x_file.header)
        pred_name = train_dict["save_folder"]+"pred/"+file_name
        nib.save(pred_file, pred_name)
        print(pred_name)
