
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

from monai.inferers import sliding_window_inference
from utils import iter_all_order

model_list = [
    ["BTCV_v1_unet",],
    ["BTCV_v2_unetr",],
    ["BTCV_v2_unetr_pre",],
]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx][0]


# for name in model_list:
test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
# test_dict["gpu_ids"] = model_list[current_model_idx][1]
test_dict["gpu_ids"] = [7]
test_dict["eval_file_cnt"] = 0
test_dict["eval_save_folder"] = "pred_monai"

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
test_dict["alt_blk_depth"] = [1,]

print("input size:", test_dict["input_size"])
# print("alt_blk_depth", test_dict["alt_blk_depth"])



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
    y_path = file_path.replace("img", "label")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    ax, ay, az = x_data.shape
    n_seg = y_data.shape[0]
    case_loss = 0

    input_data = np.expand_dims(x_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)

    order_list = iter_all_order(test_dict["alt_blk_depth"])
    # order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
    order_list_cnt = len(order_list)
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
                    # is_WDO=model_list[current_model_idx][-1],
                    )
        y_hat = nn.Softmax(dim=1)(y_hat).cpu().detach().numpy()
        # print(y_hat.shape)
        y_hat = np.argmax(np.squeeze(y_hat), axis=0)
        output_array[idx_es, :, :, :] = y_hat

    output_data = np.median(output_array, axis=0)
    # output_std = np.std(output_array, axis=0)
    # output_mean = np.mean(output_array, axis=0)
    # output_cov = np.divide(output_std, output_mean+1e-12)
    # print(output_data.shape)

    test_file = nib.Nifti1Image(np.squeeze(output_data), y_file.affine, y_file.header)
    test_save_name = test_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name
    nib.save(test_file, test_save_name)
    print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_std), x_file.affine, x_file.header)
    # test_save_name = test_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_std.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_mean), x_file.affine, x_file.header)
    # test_save_name = test_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_mean.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)
