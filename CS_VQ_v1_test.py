
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
from monai.inferers import sliding_window_inference
import bnn

# from utils import add_noise, weighted_L1Loss
# from model import UNet_Theseus as UNet
from model import VQ2d_v1
from utils import add_noise

model_list = [
    ["v1_Gau050", "Gaussian", (0, 0.5)],
    ["v2_Gau025", "Gaussian", (0, 0.25)],
    ["v3_Poi100", "Poisson", (1,)],
    ["v4_Poi025", "Poisson", (0.25,)],
    ["v5_S&P025", "Salt&Pepper", (0.975, 0.025)],
    ["v6_S&P050", "Salt&Pepper", (0.95, 0.05)],
    ["v7_SPK025", "Speckle", (0, 0.25)],
    ["v8_SPK050", "Speckle", (0, 0.5)],
    ["v9_RIC005", "Racian", (5,)],
    ["v10_RIC010", "Racian", (10,)],
    ["v11_RAY005", "Rayleigh", (5, )],
    ["v12_RAY010", "Rayleigh", (10, )],
    ["v13_RIC015", "Racian", (15,)],
    ["v14_RAY015", "Rayleigh", (15, )],
    ["v15_RIC020", "Racian", (20,)],
    ["v116_RAY020", "Rayleigh", (20, )],
    ["v15_RIC025", "Racian", (25,)],
    ["v116_RAY025", "Rayleigh", (25, )],
    ]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)

# current_model_idx = 0
# ==================== dict and config ====================

train_dict = {}

train_dict["pred_noise_folder"] = model_list[current_model_idx][0]
train_dict["noise_type"] = model_list[current_model_idx][1]
train_dict["noise_params"] = model_list[current_model_idx][2]

train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "CSVQ_v1_0102"
train_dict["gpu_ids"] = [0,]

train_dict["dropout"] = 0.
train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["alpha_dropout_consistency"] = 1

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["split_JSON"] = "./data_dir/CS_VQ_v1.json"
train_dict["pred_folder"] = train_dict["pred_noise_folder"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
train_dict["epochs"] = 5000
train_dict["batch"] = 32

train_dict["model_term"] = "VQ2d_v1"
train_dict["dataset_ratio"] = 1
train_dict["test_case_num"] = 5
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False
train_dict["data_variance"] = 1

for path in [train_dict["save_folder"], train_dict["save_folder"]+train_dict["pred_folder"]]:
    if not os.path.exists(path):
        os.mkdir(path)

# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = VQ2d_v1(
#     img_channels = model_dict["img_channels"], 
#     num_hiddens = model_dict["num_hiddens"], 
#     num_residual_layers = model_dict["num_residual_layers"], 
#     num_residual_hiddens = model_dict["num_residual_hiddens"], 
#     num_embeddings = model_dict["num_embeddings"], 
#     embedding_dim = model_dict["embedding_dim"], 
#     commitment_cost = model_dict["commitment_cost"], 
#     decay = model_dict["decay"])

model_list = sorted(glob.glob(os.path.join(train_dict["save_folder"], "model_best_*.pth")))
if "latest" in model_list[-1]:
    print("Remove model_best_latest")
    model_list.pop()
target_model = model_list[-1]
# target_model = test_dict["save_folder"]+test_dict["best_model_name"]
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")
model = model.to(device)

loss_func = torch.nn.SmoothL1Loss()

# ==================== data division ====================

from monai.data import (
    load_decathlon_datalist,
)

root_dir = train_dict["save_folder"]
split_JSON = train_dict["split_JSON"]
print("root_dir: ", root_dir)
print("split_JSON: ", split_JSON)
test_list = load_decathlon_datalist(split_JSON, False, "test", base_dir = "./")
if train_dict["test_case_num"] > 0:
    test_list = test_list[:train_dict["test_case_num"]]
# ==================== training ====================

total_test_batch = len(test_list)

# test
model.eval()
test_loss = np.zeros((total_test_batch, 3))
for test_idx, test_path_dict in enumerate(test_list):
    test_path = test_path_dict["image"]
    print(" ^Test^ ===> Case[{:03d}]/[{:03d}]: -->".format(
        test_idx+1, total_test_batch), "<--", end="")

    input_file = nib.load(test_path)
    input_data = input_file.get_fdata()
    input_data = add_noise(
        x = input_data, 
        noise_type = train_dict["noise_type"],
        noise_params = train_dict["noise_params"],
        )
    output_data = np.zeros(input_data.shape)

    cnt_zslice = input_data.shape[2]
    test_loss = np.zeros((cnt_zslice, 3))

    for idx_z in range(cnt_zslice):
        input_tensor = np.expand_dims(np.squeeze(input_data[:, :, idx_z]), (0,1))
        input_tensor = torch.from_numpy(input_tensor).float().to(device)
        with torch.no_grad():
            vq_loss, mr_recon, perplexity = model(input_tensor)
            loss_recon = loss_func(input_tensor, mr_recon) / train_dict["data_variance"]

        output_data[:, :, idx_z] = mr_recon.cpu().detach().numpy()
        test_loss[idx_z, 0] = vq_loss.item()
        test_loss[idx_z, 1] = loss_recon.item()
        test_loss[idx_z, 2] = perplexity.cpu().detach().numpy()

    print(" ^VQ : ", np.mean(test_loss[:, 0]), end="")
    print(" ^Recon: ", np.mean(test_loss[:, 1]), end="")
    print(" ^Plex: ", np.mean(test_loss[:, 2]))
    np.save(train_dict["save_folder"]+"pred/case_loss_test_{}.npy".format(
        os.path.basename(test_path)[:-7]), test_loss)

    output_file = nib.Nifti1Image(np.squeeze(input_data), input_file.affine, input_file.header)
    output_savename = train_dict["save_folder"]+train_dict["pred_folder"]+os.path.basename(test_path).replace(".nii.gz", "_noise.nii.gz")
    nib.save(output_file, output_savename)
    print(output_savename)

    output_file = nib.Nifti1Image(np.squeeze(output_data), input_file.affine, input_file.header)
    output_savename = train_dict["save_folder"]+train_dict["pred_folder"]+os.path.basename(test_path)
    nib.save(output_file, output_savename)
    print(output_savename)
