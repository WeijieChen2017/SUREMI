
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
    ["CSVQ_v3_2d_embd128", [0]],
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
model_dict["num_embeddings"] = 128
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

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = VQ2d_v1(
    img_channels = model_dict["img_channels"], 
    num_hiddens = model_dict["num_hiddens"], 
    num_residual_layers = model_dict["num_residual_layers"], 
    num_residual_hiddens = model_dict["num_residual_hiddens"], 
    num_embeddings = model_dict["num_embeddings"], 
    embedding_dim = model_dict["embedding_dim"], 
    commitment_cost = model_dict["commitment_cost"], 
    decay = model_dict["decay"])

model.train()
model = model.to(device)

loss_func = torch.nn.SmoothL1Loss()
# loss_doc = torch.nn.SmoothL1Loss()

optim = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

from monai.config import print_config
from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureChannelFirstd,
    SqueezeDimd,
)
from monai.data import (
    Dataset,
    DataLoader,
    load_decathlon_datalist,
    PatchIterd,
    GridPatchDataset,
)

print_config()

train_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        RandSpatialCropd(
            keys="image",
            roi_size=(256, 256, 1),
            random_size=False,
            ),
        SqueezeDimd(
            keys="image", dim=-1
        ),  # squeeze the last dim
        RandFlipd(
            keys="image",
            spatial_axis=[0],
            prob=0.25,
        ),
        RandFlipd(
            keys="image",
            spatial_axis=[1],
            prob=0.25,
        ),
        # RandFlipd(
        #     keys="image",
        #     spatial_axis=[2],
        #     prob=0.25,
        # ),
        RandRotate90d(
            keys="image",
            prob=0.25,
            max_k=2,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        RandSpatialCropd(
            keys="image",
            roi_size=(256, 256, 1),
            random_size=False,
            ),
        SqueezeDimd(
            keys="image", dim=-1
        ),  # squeeze the last dim
    ]
)

root_dir = train_dict["save_folder"]
split_JSON = train_dict["split_JSON"]
print("root_dir: ", root_dir)
print("split_JSON: ", split_JSON)
train_list = load_decathlon_datalist(split_JSON, False, "training", base_dir = "./")
val_list = load_decathlon_datalist(split_JSON, False, "val", base_dir = "./")

train_ds = Dataset(
    data = train_list,
    transform = train_transforms,
)

val_ds = Dataset(
    data = val_list,
    transform = val_transforms,
)

train_loader = DataLoader(
    train_ds, batch_size=train_dict["batch"], shuffle=True, 
    num_workers=8, pin_memory=True,
)

val_loader = DataLoader(
    val_ds, batch_size=train_dict["batch"], shuffle=True, 
    num_workers=4, pin_memory=True,
)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0

global_step_curr = 0
total_train_batch = len(train_loader)
total_val_batch = len(val_loader)
# epoch_loss = np.zeros((train_dict["epochs"], 2))

package_train = [train_list, True, False, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for global_step_curr in range(train_dict["epochs"]):
    global_step = global_step_curr + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(global_step+1))

    # train
    model.train()
    train_loss = np.zeros((total_train_batch, 3))
    for train_step, batch in enumerate(train_loader):
        print(" ^Train^ ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: -->".format(
                global_step+1, train_step+1, total_train_batch), "<--", end="")
        
        mr_hq = batch["image"]
        mr_hq = add_noise(
            x = mr_hq, 
            noise_type = "Racian",
            noise_params = (20,),
            ).to(device)
        # mr_hq = batch["image"].cuda()
        optim.zero_grad()
        vq_loss, mr_recon, perplexity = model(mr_hq)
        loss_recon = loss_func(mr_hq, mr_recon) / train_dict["data_variance"]
        loss = loss_recon + vq_loss
        loss.backward()
        optim.step()
        train_loss[train_step, 0] = vq_loss.item()
        train_loss[train_step, 1] = loss_recon.item()
        train_loss[train_step, 2] = perplexity
        print(" VQ_loss: ", train_loss[train_step, 0], 
              " Recon: ", train_loss[train_step, 1],
              " Perplexity: ", train_loss[train_step, 2])

    print(" ^Train^ ===>===> Epoch[{:03d}]: ".format(global_step+1), end='')
    print(" ^VQ : ", np.mean(train_loss[:, 0]), end="")
    print(" ^Recon: ", np.mean(train_loss[:, 1]), end="")
    print(" ^Plex: ", np.mean(train_loss[:, 2]))
    np.save(train_dict["save_folder"]+"loss/epoch_loss_train_{:03d}.npy".format(global_step+1), train_loss)


    # 2d validation
    model.eval()
    val_loss = np.zeros((total_val_batch, 3))
    for val_step, batch in enumerate(val_loader):
        print(" ^Val^ ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: -->".format(
                global_step+1, val_step+1, total_val_batch), "<--", end="")
        mr_hq = batch["image"].cuda()
        with torch.no_grad():
            vq_loss, mr_recon, perplexity = model(mr_hq)
            loss_recon = loss_func(mr_hq, mr_recon) / train_dict["data_variance"]

        val_loss[val_step, 0] = vq_loss.item()
        val_loss[val_step, 1] = loss_recon.item()
        val_loss[val_step, 2] = perplexity
        print(" VQ_loss: ", val_loss[val_step, 0], 
              " Recon: ", val_loss[val_step, 1],
              " Perplexity: ", val_loss[val_step, 2])

    print(" ^Val^ ===>===> Epoch[{:03d}]: ".format(global_step+1), end='')
    print(" ^VQ : ", np.mean(val_loss[:, 0]), end="")
    print(" ^Recon: ", np.mean(val_loss[:, 1]), end="")
    print(" ^Plex: ", np.mean(val_loss[:, 2]))
    np.save(train_dict["save_folder"]+"loss/epoch_loss_val_{:03d}.npy".format(global_step+1), val_loss)
    
    torch.save(model, train_dict["save_folder"]+"model_latest.pth")
    if np.mean(val_loss[:, 1]) < best_val_loss:
        # save the best model
        torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(global_step + 1))
        torch.save(optim, train_dict["save_folder"]+"optim_{:03d}.pth".format(global_step + 1))
        print("Checkpoint saved at Epoch {:03d}".format(global_step + 1))
        best_val_loss = np.mean(val_loss[:, 1])
        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()

