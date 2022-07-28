
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

# from monai.networks.nets.unet import UNet
from monai.networks.layers.factories import Act, Norm
from monai.inferers import sliding_window_inference
import bnn

from model import UNet_MDO as UNet
from model import UNet_Blockwise
from utils import iter_all_order

model_list = [
    # "Theseus_v2_181_200_rdp0",
    # "Theseus_v2_181_200_rdp1",
    # "Theseus_v2_181_200_rdp020",
    # "Theseus_v2_181_200_rdp040",
    # "Theseus_v2_181_200_rdp060",
    # "Theseus_v2_181_200_rdp080",
    "Theseus_v2_47_57_rdp000",
    "Theseus_v2_47_57_rdp020",
    "Theseus_v2_47_57_rdp040",
    "Theseus_v2_47_57_rdp060",
    "Theseus_v2_47_57_rdp080",
    "Theseus_v2_47_57_rdp100",
]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx]


# for name in model_list:
test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name # "Bayesian_MTGD_v2_unet_do10_MTGD15"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = [2]
test_dict["eval_file_cnt"] = 0
test_dict["batch"] = 8
# test_dict["best_model_name"] = "model_best_193.pth"
# test_dict["eval_sample"] = 100
test_dict["eval_save_folder"] = "pred_monai"

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
# test_dict["alt_blk_depth"] = train_dict["model_para"]["macro_dropout"]
test_dict["alt_blk_depth"] = [2,2,2,2,2,2,2]

print("input size:", test_dict["input_size"])
print("alt_blk_depth", test_dict["alt_blk_depth"])


unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["out_channels"] = 1
unet_dict["channels"] = (32, 64, 128, 256)
unet_dict["strides"] = (2, 2, 2)
unet_dict["num_res_units"] = 4
unet_dict["act"] = Act.PRELU
unet_dict["normunet"] = Norm.INSTANCE
unet_dict["dropout"] = 0.
unet_dict["bias"] = True


for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict_blk.npy", test_dict)


# ==================== basic settings ====================

np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
if "curr" in model_list[-1]:
    print("Remove model_best_curr")
    model_list.pop()
target_model = model_list[-1]
# target_model = test_dict["save_folder"]+test_dict["best_model_name"]
model_pre = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

model = UNet_Blockwise( 
    spatial_dims=unet_dict["spatial_dims"],
    in_channels=unet_dict["in_channels"],
    out_channels=unet_dict["out_channels"],
    channels=unet_dict["channels"],
    strides=unet_dict["strides"],
    num_res_units=unet_dict["num_res_units"],
    act=unet_dict["act"],
    norm=unet_dict["normunet"],
    dropout=unet_dict["dropout"],
    bias=unet_dict["bias"],
    )

pre_state = model_pre.state_dict()
model_state_keys = model.state_dict().keys()
new_model_state = {}

for model_key in model_state_keys:
    new_model_state[model_key] = pre_state[model_key]
    
model.load_state_dict(new_model_state)

model.eval()
model = model.to(device)

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X']
if test_dict["eval_file_cnt"] > 0:
    X_list = X_list[:test_dict["eval_file_cnt"]]
X_list.sort()

# ==================== fm array ====================

set_feature_map = [
# np.zeros((1, test_dict["batch"], 1, 96, 96, 96)),
np.zeros((2, test_dict["batch"], 32, 48, 48, 48)),
np.zeros((4, test_dict["batch"], 64, 24, 24, 24)),
np.zeros((8, test_dict["batch"], 128, 12, 12, 12)),
np.zeros((16, test_dict["batch"], 256, 12, 12, 12)),
np.zeros((32, test_dict["batch"], 64, 24, 24, 24)),
np.zeros((64, test_dict["batch"], 32, 48, 48, 48)),
np.zeros((128, test_dict["batch"], 1, 96, 96, 96)),
]

# ==================== training ====================

file_list = X_list
iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    y_path = file_path.replace("x", "y")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    ax, ay, az = x_data.shape
    case_loss = 0

    batch_x = np.zeros((test_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
    # batch_y = np.zeros((test_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

    for idx_batch in range(test_dict["batch"]):
        
        d0_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][1])
        d1_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][2])
        d2_offset = np.random.randint(x_data.shape[2] - train_dict["input_size"][0])

        x_slice = x_data[d0_offset:d0_offset+train_dict["input_size"][0],
                         d1_offset:d1_offset+train_dict["input_size"][1],
                         d2_offset:d2_offset+train_dict["input_size"][2]
                         ]
        # y_slice = y_data[d0_offset:d0_offset+train_dict["input_size"][0],
        #                  d1_offset:d1_offset+train_dict["input_size"][1],
        #                  d2_offset:d2_offset+train_dict["input_size"][2]
        #                  ]
        batch_x[idx_batch, 0, :, :, :] = x_slice
        # batch_y[idx_batch, 0, :, :, :] = y_slice

    batch_x = torch.from_numpy(batch_x).float().to(device)
    # batch_y = torch.from_numpy(batch_y).float().to(device)

    # set_feature_map[0][0, :, :, :, :, :] = batch_x

    ans_blk_1 = model(x=batch_x, block_idx=1)
    set_feature_map[0][0, :, :, :, :, :] = ans_blk_1[0].cpu().detach().numpy()
    set_feature_map[0][1, :, :, :, :, :] = ans_blk_1[1].cpu().detach().numpy()
    
    for alt_idx in range(4):
        block_idx = alt_idx + 1
        cnt_input = 0
        for idx_x in range(set_feature_map[alt_idx].shape[0]):
            input_x = np.squeeze(set_feature_map[alt_idx][idx_x, :, :, :, :, :])
            input_x = torch.from_numpy(input_x).float().to(device)
            ans_blk = model(x=input_x, block_idx=block_idx+1)
            set_feature_map[block_idx][cnt_input, :, :, :, :, :] = ans_blk[0].cpu().detach().numpy()
            set_feature_map[block_idx][cnt_input+1, :, :, :, :, :] = ans_blk[1].cpu().detach().numpy()
            cnt_input += 2
            print(cnt_input)



    # output_data = np.median(output_array, axis=0)
    # output_std = np.std(output_array, axis=0)
    # output_mean = np.mean(output_array, axis=0)
    # # output_cov = np.divide(output_std, output_mean+1e-12)
    # print(output_data.shape)

    # test_file = nib.Nifti1Image(np.squeeze(output_data), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name
    # nib.save(test_file, test_save_name)
    # print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_std), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_std.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_mean), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_mean.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)
