
# Generative Confidential Network

import os
import gc
import copy
import glob
import time
# import wandb
import random
import numpy as np


model_list = [
    # ["syn_DLE_4444111", [7], [4,4,4,4,1,1,1], [[1, 0], [1, 2], [0, 2], [3, 1], [], [], []]],
    # ["syn_DLE_1114444", [7], [1,1,1,4,4,4,4], [[], [], [], [2, 1], [1, 0], [3, 1], [2, 0]]],
    # ["syn_DLE_4444444", [7], [4,4,4,4,4,4,4], [[1, 2], [1, 3], [3, 0], [3, 0], [1, 3], [3, 0], [2, 3]]],
    # ["syn_DLE_2222222_e400_lrn4", [5], [2,2,2,2,2,2,2], [[], [], [], [], [], [], []], 0],
    # ["syn_DLE_1114444_e400_lrn4", [5], [1,1,1,4,4,4,4], [[], [], [], [], [], [], []], 0],
    # ["syn_DLE_4444111_e400_lrn4", [5], [4,4,4,4,1,1,1], [[], [], [], [], [], [], []], 178],
    # ["syn_DLE_4444444_e400_lrn4", [5], [4,4,4,4,4,4,4], [[], [], [], [], [], [], []], 0],
#     ["syn_DLE_2222222_e400_lrn4", [5], [2,2,2,2,2,2,2], [[], [], [], [], [], [], []], 0],
#     ["syn_DLE_1114444_e400_lrn4", [5], [1,1,1,4,4,4,4], [[], [], [], [2, 3], [3, 0], [2, 3], [1, 3]], 0],
#     ["syn_DLE_4444111_e400_lrn4", [5], [4,4,4,4,1,1,1], [[1, 2], [1, 0], [3, 2], [3, 1], [], [], []], 0],
#     ["syn_DLE_4444444_e400_lrn4", [5], [4,4,4,4,4,4,4], [[0, 3], [0, 1], [1, 0], [3, 0], [1, 0], [0, 2], [0, 1]], 0],
    # ["syn_DLE_2222222_e400_lrn4", [3], [2,2,2,2,2,2,2], [[], [], [], [], [], [], []], 0],
    ["syn_DLE_1114444_e400_lrn4", [0], [1,1,1,4,4,4,4], [[], [], [], [], [], [], []], 0],
    ["syn_DLE_4444111_e400_lrn4", [0], [4,4,4,4,1,1,1], [[], [], [], [], [], [], []], 0],
    ["syn_DLE_4444444_e400_lrn4", [0], [4,4,4,4,4,4,4], [[], [], [], [], [], [], []], 0],
]




print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

name = model_list[current_model_idx][0]
gpu_list = model_list[current_model_idx][1]
alt_block_num = model_list[current_model_idx][2]
block_kickout = model_list[current_model_idx][3]
# block_kickout = []
eval_start = model_list[current_model_idx][4]


# for name in model_list:
test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = name # "Bayesian_MTGD_v2_unet_do10_MTGD15"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = gpu_list
test_dict["eval_file_cnt"] = 0
# test_dict["best_model_name"] = "model_best_193.pth"
# test_dict["eval_sample"] = 100
test_dict["eval_save_folder"] = "full_val_xte"
test_dict["special_cases"] = ["00522"]
test_dict["eval_start"] = eval_start

test_dict["save_tag"] = ""

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
# test_dict["alt_blk_depth"] = train_dict["model_para"]["macro_dropout"]
test_dict["alt_blk_depth"] = alt_block_num

print("input size:", test_dict["input_size"])
print("alt_blk_depth", test_dict["alt_blk_depth"])



for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)


np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

# import torch
import torchvision
import requests

# from monai.networks.nets.unet import UNet
from monai.networks.layers.factories import Act, Norm
from monai.inferers import sliding_window_inference
from scipy.stats import zscore
# import bnn

from model import UNet_MDO as UNet
from utils import iter_all_order, iter_some_order, iter_all_order_but




# ==================== basic settings ====================

model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
if "curr" in model_list[-1]:
    print("Remove model_best_curr")
    model_list.pop()
target_model = model_list[-1]
print(target_model)
# target_model = test_dict["save_folder"]+test_dict["best_model_name"]
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")
# print(model)
# exit()

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X']
# X_list = data_div['val_list_X']
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

if test_dict["eval_start"] > 0:
    file_list = file_list[test_dict["eval_start"]:]

iter_tag = "test"
cnt_total_file = len(file_list)
cnt_each_cube = 1
model.eval()
model = model.to(device)

# order_list, _ = iter_some_order(test_dict["alt_blk_depth"], order_need=128)

if block_kickout == []:
    order_list, _ = iter_all_order(test_dict["alt_blk_depth"])
    if len(order_list) > 128:
        order_list, _ = iter_some_order(test_dict["alt_blk_depth"], order_need=128)
    # order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
else:
    order_list, _ = iter_all_order_but(test_dict["alt_blk_depth"], remove_blocks=block_kickout)
    print("Bad blocks have been kicked out!")
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

    ax, ay, az = x_data.shape
    case_loss = 0

    input_data = x_data
    # input_data = np.pad(x_data, ((96,96),(96,96),(96,96)), 'constant')
    input_data = np.expand_dims(input_data, (0,1))
    input_data = torch.from_numpy(input_data).float().to(device)
    output_array = np.zeros((order_list_cnt, ax, ay, az))

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
            mae_error = np.mean(np.absolute(curr_pred-y_data))*4000
            for alt_num in range(len(alt_block_num)):
                error_vote[alt_num][order_list[idx_es][alt_num]].append(mae_error)
            output_array[idx_es, :, :, :] = np.squeeze(y_hat.cpu().detach().numpy())

    output_data = np.median(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    # output_mean = np.mean(output_array, axis=0)
    # output_cov = np.divide(output_std, output_mean+1e-12)
    # print(output_data.shape)


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
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+".nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    test_file = nib.Nifti1Image(np.squeeze(output_std), x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_std.nii.gz")
    nib.save(test_file, test_save_name)
    print(test_save_name)

    error_vote_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/error_vote_"+file_name.replace(".nii.gz", test_dict["save_tag"]+".npy")
    np.save(error_vote_name, error_vote)
    print(error_vote_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_mean), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_mean.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_norm_std), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_norm_std.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)

    # test_file = nib.Nifti1Image(np.squeeze(output_z_std), x_file.affine, x_file.header)
    # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", test_dict["save_tag"]+"_z_std.nii.gz")
    # nib.save(test_file, test_save_name)
    # print(test_save_name)
