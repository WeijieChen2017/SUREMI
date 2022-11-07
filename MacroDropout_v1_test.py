
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
from utils import iter_all_order

model_list = [
    # "MDO_v1_222222222",
    # "MDO_v2_333333333",
    # "MDO_v3_224484422",
    # "MDO_v4_884424488",
    # "Bayesian_unet_v12_unet_drop10",
    # "Bayesian_unet_v13_unet_drop25",
    # "Bayesian_unet_v14_unet_drop50",
    # "Bayesian_unet_v15_unet_drop75",
    ["RDO_v1_R00001_D50", [0]],
    ["RDO_v1_R00010_D50", [0]],
    ["RDO_v1_R00100_D25", [0]],
    ["RDO_v1_R00100_D50", [3]],
    ["RDO_v1_R00100_D75", [3]],
    ["RDO_v2_dim3_R100_D50", [4]],
    ["RDO_v2_dim3_R050_D50", [4]],
    ["RDO_v2_dim3_R100_D25", [5]],
    ["RDO_v2_dim3_R100_D75", [5]],
]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx][0]
gpu_list = model_list[current_model_idx][1]

# for name in model_list:
test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name # "Bayesian_MTGD_v2_unet_do10_MTGD15"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = gpu_list
test_dict["eval_file_cnt"] = 0
# test_dict["best_model_name"] = "model_best_193.pth"
test_dict["eval_sample"] = 128
test_dict["eval_save_folder"] = "pred_monai"

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]
print("input size:", train_dict["input_size"])
# print("alt_blk_depth", train_dict["model_para"]["macro_dropout"])

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
# test_dict["alt_blk_depth"] = train_dict["model_para"]["macro_dropout"]


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

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X']
if test_dict["eval_file_cnt"] > 0:
    X_list = X_list[:test_dict["eval_file_cnt"]]
X_list.sort()


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

    input_data = np.expand_dims(x_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)

    # order_list = iter_all_order(test_dict["alt_blk_depth"])
    # order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
    # order_list_cnt = len(order_list)
    order_list_cnt = test_dict["eval_sample"]
    output_array = np.zeros((order_list_cnt, ax, ay, az))

    for idx_es in range(order_list_cnt):
        with torch.no_grad():
            # print(order_list[idx_es])
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = test_dict["input_size"], 
                    sw_batch_size = 16, 
                    predictor = model,
                    overlap=0.25, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    # order=order_list[idx_es],
                    )
            output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()

    output_data = np.median(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    output_mean = np.mean(output_array, axis=0)
    # output_cov = np.divide(output_std, output_mean+1e-6)
    print(output_data.shape)

    test_file = nib.Nifti1Image(np.squeeze(output_data), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_file = nib.Nifti1Image(np.squeeze(output_std), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_std.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_file = nib.Nifti1Image(np.squeeze(output_mean), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_mean.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)
