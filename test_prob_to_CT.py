import os
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
from sklearn import cluster
import torch.nn as nn

import torch
import torchvision
import requests

from model import UNet, UNet_seg

def bin_CT(img, n_bins=128):
    data_vector = img
    data_max = np.amax(data_vector)
    data_min = np.amin(data_vector)
    data_squeezed = (data_vector-data_min)/(data_max-data_min)
    data_extended = data_squeezed * (n_bins-1)
    data_discrete = data_extended // 1
    return np.asarray(list(data_discrete), dtype=np.int64)
# ==================== dict and config ====================

test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = "Prob_to_CT"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["seed"] = 426
test_dict["input_channel"] = 5
test_dict["output_channel"] = 1
test_dict["gpu_ids"] = [5]
test_dict["epochs"] = 50
test_dict["batch"] = 10
test_dict["dropout"] = 0
test_dict["model_term"] = "UNet_seg"
test_dict["model_save_name"] = "model_best_011.pth"

test_dict["folder_X"] = "./data_dir/T1_T2/"
# test_dict["folder_Y"] = "./data_dir/norm_CT/regular/"
test_dict["val_ratio"] = 0.3
test_dict["test_ratio"] = 0.2

test_dict["loss_term"] = "SmoothL1Loss"
test_dict["optimizer"] = "AdamW"
test_dict["opt_lr"] = 1e-3 # default
test_dict["opt_betas"] = (0.9, 0.999) # default
test_dict["opt_eps"] = 1e-8 # default
test_dict["opt_weight_decay"] = 0.01 # default
test_dict["amsgrad"] = False # default

for path in [test_dict["save_folder"], test_dict["save_folder"]+"npy/", test_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

# wandb.init(project=test_dict["project_name"])
# config = wandb.config
# config.in_chan = test_dict["input_channel"]
# config.out_chan = test_dict["output_channel"]
# config.epochs = test_dict["epochs"]
# config.batch = test_dict["batch"]
# config.dropout = test_dict["dropout"]
# config.moodel_term = test_dict["model_term"]
# config.loss_term = test_dict["loss_term"]
# config.opt_lr = test_dict["opt_lr"]
# config.opt_weight_decay = test_dict["opt_weight_decay"]

np.save(test_dict["save_folder"]+"_test_dict.npy", test_dict)

# ==================== basic settings ====================

np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = torch.load(test_dict["save_folder"]+test_dict["model_save_name"])
print("The model loaded from ", test_dict["save_folder"]+test_dict["model_save_name"])
# model = UNet_seg(n_channels=test_dict["input_channel"], n_classes=test_dict["output_channel"])
model.eval().float()
model = model.to(device)
criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = test_dict["opt_lr"],
    betas = test_dict["opt_betas"],
    eps = test_dict["opt_eps"],
    weight_decay = test_dict["opt_weight_decay"],
    amsgrad = test_dict["amsgrad"]
    )

# ==================== data division ====================

X_list = sorted(glob.glob(test_dict["folder_X"]+"*.nii.gz"))

# ==================== training ====================

for cnt_file, file_path in enumerate(X_list):
    
    file_name = os.path.basename(file_path)
    cube_x_path = file_path
    print("--->",cube_x_path,"<---", end="")
    file_x = nib.load(cube_x_path)
    cube_x_data = file_x.get_fdata()
    pred_x_data = np.zeros((cube_x_data.shape))
    len_z = cube_x_data.shape[2]
    input_list = list(range(len_z-2))
    
    for idx_iter in range(len_z//test_dict["batch"]):

        batch_x = np.zeros((test_dict["batch"], test_dict["input_channel"], cube_x_data.shape[0], cube_x_data.shape[1]))

        res = cube_x_data.shape[0]
        nX_clusters = test_dict["input_channel"]

        for idx_batch in range(test_dict["batch"]):
            z_center = input_list[idx_iter*test_dict["batch"]+idx_batch] + 1
            X_data = bin_CT(cube_x_data[:, :, z_center-1:z_center+2])
            X_cluster = cluster.KMeans(n_clusters=nX_clusters)
            X_flatten = np.reshape(X_data, (res*res, 3))
            X_flatten_k = X_cluster.fit_predict(X_flatten)
            X_data_k = np.reshape(X_flatten_k, (res, res))
            unique, counts = np.unique(X_flatten_k, return_counts=True)
            max_elem_count = np.amax(counts)
            max_elem_label = np.where(counts==np.amax(counts))[0][0]

            # set background (max label elem) to 0
            X_flatten_k[X_flatten_k == max_elem_label] = 10
            X_flatten_k[X_flatten_k == 0] = max_elem_label
            X_flatten_k[X_flatten_k == 10] = 0
            
            for idx_cs in range(nX_clusters):
                X_iso_slice = np.zeros((res, res))
                X_mask = np.asarray([X_flatten_k == int(idx_cs)]).reshape((res, res))
                X_iso_slice[X_mask] = 1
                batch_x[idx_batch, idx_cs, :, :] = X_iso_slice
                        
        batch_x = torch.from_numpy(batch_x).float().to(device)

        optimizer.zero_grad()
        y_hat = model(batch_x).detach().cpu().numpy()

        for idx_batch in range(test_dict["batch"]):
            z_center = input_list[idx_iter*test_dict["batch"]+idx_batch] + 1
            pred_x_data[:, :, z_center] = y_hat[idx_batch, 0, :, :]
    
    pred_file = nib.Nifti1Image(pred_x_data, file_x.affine, file_x.header)
    pred_name = os.path.dirname(file_path) + "pred_" + file_name
    nib.save(pred_file, pred_name)
    print(file_path)


