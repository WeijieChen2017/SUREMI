import os
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
from sklearn import cluster
# import torch.nn as nn

# import torch
# import torchvision
# import requests

# from model import UNet, UNet_seg

def bin_CT(img, n_bins=128):
    data_vector = img
    data_max = np.amax(data_vector)
    data_min = np.amin(data_vector)
    data_squeezed = (data_vector-data_min)/(data_max-data_min)
    data_extended = data_squeezed * (n_bins-1)
    data_discrete = data_extended // 1
    return np.asarray(list(data_discrete), dtype=np.int64)
# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "MR_to_seg"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_channel"] = 5
train_dict["output_channel"] = 1
train_dict["gpu_ids"] = [5]
train_dict["epochs"] = 50
train_dict["batch"] = 10
train_dict["dropout"] = 0
train_dict["model_term"] = "UNet_seg"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

# wandb.init(project=train_dict["project_name"])
# config = wandb.config
# config.in_chan = train_dict["input_channel"]
# config.out_chan = train_dict["output_channel"]
# config.epochs = train_dict["epochs"]
# config.batch = train_dict["batch"]
# config.dropout = train_dict["dropout"]
# config.moodel_term = train_dict["model_term"]
# config.loss_term = train_dict["loss_term"]
# config.opt_lr = train_dict["opt_lr"]
# config.opt_weight_decay = train_dict["opt_weight_decay"]

np.save(train_dict["save_folder"]+"dict.npy", train_dict)