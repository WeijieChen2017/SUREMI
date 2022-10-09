
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
from scipy.stats import zscore
# import bnn

from model import UNet_MDO as UNet
from utils import iter_all_order, find_label_diff


model_list = [
    # "Theseus_v2_181_200_rdp0",
    # "Theseus_v2_181_200_rdp1",
    "Theseus_v2_181_200_rdp020",
    # "Theseus_v2_181_200_rdp040",
    # "Theseus_v2_181_200_rdp060",
    # "Theseus_v2_181_200_rdp080",
    # "Theseus_v2_47_57_rdp000",
    # "Theseus_v2_47_57_rdp020",
    # "Theseus_v2_47_57_rdp040",
    # "Theseus_v2_47_57_rdp060",
    # "Theseus_v2_47_57_rdp080",
    # "Theseus_v2_47_57_rdp100",
    # "Theseus_v6_pad",
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
test_dict["gpu_ids"] = [4]
test_dict["eval_file_cnt"] = 0
# test_dict["best_model_name"] = "model_best_193.pth"
# test_dict["eval_sample"] = 100
test_dict["eval_save_folder"] = "dgx2"
test_dict["special_cases"] = [
    # "03773",
    # "05628",
]
# test_dict["save_tag"] = ""
test_dict["save_tag"] = "_srd8_edge"
test_dict["stride_division"] = 8

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
# test_dict["alt_blk_depth"] = train_dict["model_para"]["macro_dropout"]
test_dict["alt_blk_depth"] = [2,2,2,2,2,2,2]

print("input size:", test_dict["input_size"])
print("alt_blk_depth", test_dict["alt_blk_depth"])



for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)


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
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")
# print(model)
# exit()

# ==================== data division ====================

# data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
# X_list = sorted(data_div['test_list_X'] + data_div["train_list_X"] + data_div["val_list_X"])
X_list = sorted(glob.glob("./data_dir/dgx2/MR_norm/norm/*.nii.gz"))
# if test_dict["eval_file_cnt"] > 0:
#     X_list = X_list[:test_dict["eval_file_cnt"]]
# X_list.sort()


# ==================== training ====================
# file_list = []
# if len(test_dict["special_cases"]) > 0:
#     for case_name in X_list:
#         for spc_case_name in test_dict["special_cases"]:
#             if spc_case_name in os.path.basename(case_name):
#                 file_list.append(case_name)
# else:
#     file_list = X_list

file_list = X_list
iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    # y_path = file_path.replace("x", "y")
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    ax, ay, az = x_data.shape
    case_loss = 0

    input_data = x_data
    # input_data = np.pad(x_data, ((96,96),(96,96),(96,96)), 'edge')
    input_data = np.expand_dims(input_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)

    order_list, _ = iter_all_order(test_dict["alt_blk_depth"])
    # order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
    order_list_cnt = len(order_list)
    output_array = np.zeros((order_list_cnt, ax, ay, az))

    for idx_es in range(order_list_cnt):
        with torch.no_grad():
            # print(order_list[idx_es])
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = test_dict["input_size"], 
                    sw_batch_size = 64, 
                    predictor = model,
                    overlap=1/test_dict["stride_division"], 
                    mode="gaussian", 
                    sigma_scale=1/test_dict["stride_division"], 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    order=order_list[idx_es],
                    )
            # output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()[:, :, 96:-96, 96:-96, 96:-96]
            output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()[:, :, :, :, :]

    output_data = np.median(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    # output_mean = np.mean(output_array, axis=0)
    # output_cov = np.divide(output_std, output_mean+1e-12)
    # print(output_data.shape)

    label_diff = find_label_diff(data_pred=output_data, data_std=output_std)
    total_unstable_voxel = np.sum(label_diff)



    # output_array[output_array>1] = 1
    # output_array[output_array<0] = 0
    # output_z = copy.deepcopy(output_array)
    # mask_air = output_array < 0.125
    # mask_bone = output_array > 0.375
    # mask_1 = output_array < 0.375
    # mask_2 = output_array > 0.125
    # mask_1 = mask_1.astype(int)
    # mask_2 = mask_2.astype(int)
    # mask_soft = (mask_1 * mask_2).astype(bool)

    # output_array[mask_air] = output_array[mask_air] * 8 # (0,0.125) >>> (0., 1.)
    # output_array[mask_soft] = output_array[mask_soft] - 0.125 # (0.125, 0.375) >>> (0., 0.250)
    # output_array[mask_soft] = output_array[mask_soft] * 4 # (0., 0.250) >>> (0., 1.)
    # output_array[mask_bone] = output_array[mask_bone] - 0.375 # (0.375, 1) >>> (0., 0.625)
    # output_array[mask_bone] = output_array[mask_bone] * 1.6 # (0., 0.625) >>> (0., 1.)
    # output_norm_std = np.std(output_array, axis=0)

    # air_mean = np.mean(output_z[mask_air])
    # air_std = np.std(output_z[mask_air])
    # soft_mean = np.mean(output_z[mask_soft])
    # soft_std = np.std(output_z[mask_soft])
    # bone_mean = np.mean(output_z[mask_bone])
    # bone_std = np.std(output_z[mask_bone])

    # output_z[mask_air] = (output_z[mask_air] - air_mean) / air_std
    # output_z[mask_soft] = (output_z[mask_soft] - soft_mean) / soft_std
    # output_z[mask_bone] = (output_z[mask_bone] - bone_mean) / bone_std
    # output_z_std = np.std(output_z, axis=0)

    test_file = nib.Nifti1Image(np.squeeze(output_data), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_pred.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_file = nib.Nifti1Image(np.squeeze(output_std), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_std.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_file = nib.Nifti1Image(np.squeeze(label_diff), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_unstable.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_sum_voxel.npy")
    np.save(test_save_name, total_unstable_voxel)
    print(test_save_name)
