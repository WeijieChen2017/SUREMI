import os
import gc
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

import torch
import torchvision
import requests

from model import ComplexTransformerGenerationModel as CTGM


# ==================== dict and config ====================

test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = "CTGM_2d_v1_SL1"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = [7]
test_dict["eval_file_cnt"] = 16

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]
print("input size:", train_dict["input_size"])

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
ax, ay = test_dict["input_size"]
cx = 32

for path in [test_dict["save_folder"], test_dict["save_folder"]+"pred/"]:
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
# target_model = test_dict["save_folder"]+"model_best_086.pth"
target_model = model_list[-1]
model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

model = model.to(device)

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X'][:test_dict["eval_file_cnt"]]


# train_dict = {}
# train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
# train_dict["project_name"] = "CTGM_2d_v1_SL1"
# train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
# train_dict["seed"] = 426
# train_dict["input_size"] = [256, 256]
# ax, ay = train_dict["input_size"]
# train_dict["gpu_ids"] = [7]
# train_dict["epochs"] = 2000
# train_dict["batch"] = 16
# train_dict["dropout"] = 0
# train_dict["model_term"] = "ComplexTransformerGenerationModel"

# train_dict["model_related"] = {}
# train_dict["model_related"]["cx"] = 32
# cx = train_dict["model_related"]["cx"]
# train_dict["model_related"]["input_dims"] = [cx**2, cx**2]
# train_dict["model_related"]["hidden_size"] = 1024
# train_dict["model_related"]["embed_dim"] = 1024
# train_dict["model_related"]["output_dim"] = cx**2*2
# train_dict["model_related"]["num_heads"] = cx
# train_dict["model_related"]["attn_dropout"] = 0.0
# train_dict["model_related"]["relu_dropout"] = 0.0
# train_dict["model_related"]["res_dropout"] = 0.0
# train_dict["model_related"]["out_dropout"] = 0.0
# train_dict["model_related"]["layers"] = 6
# train_dict["model_related"]["attn_mask"] = False

# train_dict["folder_X"] = "./data_dir/Iman_MR/kspace_2d/"
# train_dict["folder_Y"] = "./data_dir/Iman_CT/kspace_2d/"
# train_dict["val_ratio"] = 0.3
# train_dict["test_ratio"] = 0.2

# train_dict["loss_term"] = "SmoothL1Loss"
# train_dict["optimizer"] = "AdamW"
# train_dict["opt_lr"] = 1e-3 # default
# train_dict["opt_betas"] = (0.9, 0.999) # default
# train_dict["opt_eps"] = 1e-8 # default
# train_dict["opt_weight_decay"] = 0.01 # default
# train_dict["amsgrad"] = False # default

# ==================== training ====================

num_vocab = (ax//cx) * (ay//cx)

file_list = X_list

model.eval()

total_loss = np.zeros((len(file_list)))

"""
x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
"""

for cnt_file, file_path in enumerate(file_list):
    
    total_file = len(file_list)
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(" ===> [{:03d}]/[{:03d}]: --->".format(cnt_file+1, total_file), file_name, "<---", end="") #
    x_data = np.load(x_path)
    y_data = np.load(y_path)
    dz = x_data.shape[0]
    az = dz // (cx*cx)
    z_list = list(range(dz))
    pred_vol = np.zeros((256, 256, az))
    cnt_z = 0
    ix = 0
    iy = 0

    for iz in range(dz):

        pred_img = np.zeros((256, 256))

        batch_x = np.zeros((1, 1, cx**2*2))
        batch_y = np.zeros((1, 1, cx**2*2))

        batch_x[0, 0, :] = x_data[iz, :, :]
        batch_y[0, 0, :] = y_data[iz, :, :]

        batch_x = torch.from_numpy(batch_x).float().to(device).contiguous()
        batch_y = torch.from_numpy(batch_y).float().to(device).contiguous()
            
        y_hat = model(batch_x, batch_y).detach().cpu().numpy()
        print(y_hat.shape)
        y_hat_real = np.squeeze(y_hat[:, :, :cx**2])
        y_hat_imag = np.squeeze(y_hat[:, :, cx**2:])

        pred_cplx = np.vectorize(complex)(y_hat_real, y_hat_imag).reshape((cx, cx))
        # print(pred_cplx.shape)
        patch = np.fft.ifftn(np.fft.ifftshift(pred_cplx))
        pred_img[ix*cx:ix*cx+cx, iy*cx:iy*cx+cx] = patch

        iy += 1

        if iy == ay // cx:
            iy = 0
            ix += 1

        if ix == ax // cx:
            ix = 0
            iy = 0
            pred_vol[:, :, cnt_z] = pred_img
            cnt_z += 1

    file_CT = nib.load("./data_dir/Iman_CT/norm/"+file_name.replace("npy", "nii.gz"))
    pred_file = nib.Nifti1Image(pred_vol, file_CT.affine, file_CT.header)
    pred_name = test_dict["save_folder"]+"pred/"+file_name.replace("npy", "nii.gz")
    nib.save(pred_file, pred_name)
