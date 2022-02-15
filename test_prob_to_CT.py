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


def generate_dist_weights(data_shape):
    dist = np.zeros(data_shape)
    len_x = data_shape[0]
    len_y = data_shape[1]
    len_z = data_shape[2]
    center = [len_x // 2, len_y // 2, len_z // 2]
    for ix in range(len_x):
        for iy in range(len_y):
            for iz in range(len_z):
                dx = np.abs(ix-center[0]) ** 2
                dy = np.abs(iy-center[1]) ** 2
                dz = np.abs(iz-center[2]) ** 2
                dist[ix, iy, iz] = np.sqrt(dx+dy+dz)
    
    return dist


def dist_kmeans(X_path, nX_clusters, dist):
    X_file = nib.load(X_path)
    X_data = bin_CT(X_file.get_fdata(), n_bin=n_bin)
    
    X_cluster = cluster.KMeans(n_clusters=nX_clusters)
    X_flatten = np.ravel(X_data)
    X_flatten = np.reshape(X_flatten, (len(X_flatten), 1))
    X_flatten_k = X_cluster.fit_predict(X_flatten)
    X_data_k = np.reshape(X_flatten_k, X_data.shape)
    
    weight_data = np.multiply(X_data_k, dist)
    scores = np.zeros((nX_clusters))
    for idx in range(nX_clusters):
        cluster_map = np.where(X_data==idx, 1, 0)
        scores[idx] = np.sum(np.multiply(cluster_map, dist)) / np.sum(cluster_map)
    idx_scores = np.argsort(scores)
    print(idx_scores, scores)
    
    for idx in range(nX_clusters):
        X_data_k[X_data_k == idx] = nX_clusters+idx
    
    for idx in range(nX_clusters):
        X_data_k[X_data_k == nX_clusters+idx] = idx_scores[idx]
    
    return X_data_k

# ==================== dict and config ====================

test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = "Prob_to_CT_k10"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["seed"] = 426
test_dict["input_channel"] = 10
test_dict["output_channel"] = 1
test_dict["gpu_ids"] = [5]
test_dict["epochs"] = 50
test_dict["batch"] = 8
test_dict["dropout"] = 0
test_dict["model_term"] = "U"
test_dict["model_save_name"] = "model_best_032.pth"

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

# ==================== test ====================

nX_clusters = 10
n_bin = 256
dist = generate_dist_weights((256, 256, 182))

for cnt_file, file_path in enumerate(X_list):
    
    print("--->",file_path,"<---", end="")
    
    X_path = file_path
    X_file = nib.load(X_path)
    X_data_k = dist_kmeans(X_path, nX_clusters, dist)
    X_save_name = X_path.replace(".nii.gz", "_k10.nii.gz")
    X_save_file = nib.Nifti1Image(X_data_k, X_file.affine, X_file.header)
    nib.save(X_save_file, X_save_name)

    file_path = X_save_name
    file_name = os.path.basename(file_path)
    cube_x_path = file_path
    file_x = nib.load(cube_x_path)
    cube_x_data = file_x.get_fdata()
    pred_x_data = np.zeros((cube_x_data.shape))
    len_z = cube_x_data.shape[2]
    input_list = list(range(len_z-2))

    for idx_iter in range(len_z//test_dict["batch"]):

        batch_x = np.zeros((test_dict["batch"], test_dict["input_channel"], cube_x_data.shape[0], cube_x_data.shape[1]))

        nX_clusters = test_dict["input_channel"]

        for idx_batch in range(test_dict["batch"]):
            z_center = input_list[idx_iter*test_dict["batch"]+idx_batch]
            
            for idx_cluster in range(test_dict["input_channel"]):
                batch_x[idx_batch, idx_cluster, :, :] = np.where(cube_x_data[:, :, z_center]==idx_cluster, 1, 0)  
        
        batch_x = torch.from_numpy(batch_x).float().to(device)

        optimizer.zero_grad()
        y_hat = model(batch_x).detach().cpu().numpy()

        for idx_batch in range(test_dict["batch"]):
            z_center = input_list[idx_iter*test_dict["batch"]+idx_batch] + 1
            pred_x_data[:, :, z_center] = y_hat[idx_batch, 0, :, :]
    
    pred_file = nib.Nifti1Image(pred_x_data, file_x.affine, file_x.header)
    pred_name = os.path.dirname(file_path) + "/pred_" + file_name
    nib.save(pred_file, pred_name)
    print(file_path)


