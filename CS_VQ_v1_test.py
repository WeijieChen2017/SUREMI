
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

model_list = [
    ["CSVQ_v1_0102", [0]],
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

train_dict["dropout"] = 0.
train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["alpha_dropout_consistency"] = 1

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["split_JSON"] = "./data_dir/CS_VQ_v1.json"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
train_dict["epochs"] = 5000
train_dict["batch"] = 32

train_dict["model_term"] = "VQ2d_v1"
train_dict["dataset_ratio"] = 1
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False
train_dict["data_variance"] = 1


model_dict = {}

model_dict["img_channels"] = 1
model_dict["num_hiddens"] = 256
model_dict["num_residual_layers"] = 4
model_dict["num_residual_hiddens"] = 128
model_dict["num_embeddings"] = 512
model_dict["embedding_dim"] = 128
model_dict["commitment_cost"] = 0.25
model_dict["decay"] = 0.99
train_dict["model_para"] = model_dict


train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

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
        test_loss[val_step, 0] = vq_loss.item()
        test_loss[val_step, 1] = loss_recon.item()
        test_loss[val_step, 2] = perplexity.cpu().detach().numpy()

    print(" ^VQ : ", np.mean(test_loss[:, 0]), end="")
    print(" ^Recon: ", np.mean(test_loss[:, 1]), end="")
    print(" ^Plex: ", np.mean(test_loss[:, 2]))
    np.save(train_dict["save_folder"]+"pred/case_loss_test_{}.npy".format(
        os.path.basename[:-7]), test_loss)

    output_file = nib.Nifti1Image(np.squeeze(output_data), input_file.affine, input_file.header)
    output_savename = train_dict["save_folder"]+"pred/"+os.path.basename
    nib.save(output_file, output_savename)
    print(output_savename)
