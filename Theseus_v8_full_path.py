
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
from utils import iter_all_order, iter_some_order, iter_all_order_but
from utils import denorm_CT, cal_rmse_mae_ssim_psnr_acut_dice, cal_mae


model_list = [
    # ["syn_DLE_4444111", [0], [4,4,4,4,1,1,1], ],
    # ["syn_DLE_1114444", [0], [1,1,1,4,4,4,4], ],
    # ["syn_DLE_4444444", [0], [4,4,4,4,4,4,4], ],
    # ["syn_DLE_4444111_e400_lrn4", [0], [4,4,4,4,1,1,1], ],
    # ["syn_DLE_1114444_e400_lrn4", [0], [1,1,1,4,4,4,4], ],
    # ["syn_DLE_4444444_e400_lrn4", [0], [4,4,4,4,4,4,4], ],
    ["syn_DLE_2222222_e400_lrn4", [0], [2,2,2,2,2,2,2], ],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx][0]
gpu_list = model_list[current_model_idx][1]
alt_block_num = model_list[current_model_idx][2]

# for name in model_list:
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = gpu_list
test_dict["eval_file_cnt"] = 0
test_dict["eval_save_folder"] = "full_metric"
test_dict["special_cases"] = []

test_dict["save_tag"] = ""

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
test_dict["alt_blk_depth"] = alt_block_num

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
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
# X_list = data_div['test_list_X']
X_list = data_div['val_list_X']
if test_dict["eval_file_cnt"] > 0:
    X_list = X_list[:test_dict["eval_file_cnt"]]
X_list.sort()


# ==================== training ====================
file_list = []
if len(test_dict["special_cases"]) > 0:
    for case_name in X_list:
        for spc_case_name in test_dict["special_cases"]:
            if spc_case_name in os.path.basename(case_name):
                file_list.append(case_name)
else:
    file_list = X_list

iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

order_list, _ = iter_all_order(test_dict["alt_blk_depth"])
print(order_list)
order_list_cnt = len(order_list)


for cnt_file, file_path in enumerate(file_list):
    
    error_vote = []
    for alt_num in alt_block_num:
        curr_alt = []
        for idx in range(alt_num):
            curr_alt.append([])
        error_vote.append(curr_alt)

    x_path = file_path
    y_path = file_path.replace("x", "y")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()
    y_data_denorm = denorm_CT(y_data)

    ax, ay, az = x_data.shape
    case_loss = 0

    input_data = x_data
    # input_data = np.pad(x_data, ((96,96),(96,96),(96,96)), 'constant')
    input_data = np.expand_dims(input_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)
    # output_array = np.zeros((order_list_cnt, ax, ay, az))
    output_metric = dict()

    for idx_es in range(order_list_cnt):
        with torch.no_grad():
            # print(order_list[idx_es])
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = test_dict["input_size"], 
                    sw_batch_size = 64, 
                    predictor = model,
                    overlap=1/8, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    order=order_list[idx_es],
                    )
            curr_pred = np.squeeze(y_hat.cpu().detach().numpy())
            curr_pred_denorm = denorm_CT(curr_pred)
            # metric_list = cal_rmse_mae_ssim_psnr_acut_dice(curr_pred_denorm, y_data_denorm)
            metric_list = cal_mae(curr_pred_denorm, y_data_denorm)
            key_name = ''.join(str(order_list[idx_es]))
            output_metric[key_name] = metric_list

    save_path = os.path.join(test_dict["save_folder"], test_dict["eval_save_folder"], file_name.replace(".nii.gz", ".npy"))
    np.save(save_path, output_metric)
