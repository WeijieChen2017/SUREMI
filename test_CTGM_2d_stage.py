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
test_dict["project_name"] = "CTGM_2d_v11_mse_layer2_e80L2"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = [4]
test_dict["eval_file_cnt"] = 16
test_dict["new_stage_folder"] = "kspace_2d_e99_S3"

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]
print("input size:", train_dict["input_size"])

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]
ax, ay = test_dict["input_size"]
cx = 32

for path in [test_dict["save_folder"]+"pred/", test_dict["save_folder"]+"stage1/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)

for path in ["./data_dir/Iman_CT/"+test_dict["new_stage_folder"]+"/"]:
    if not os.path.exists(path):
        os.mkdir(path)

# ==================== basic settings ====================

np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
# if "curr" in model_list[-1]:
#     print("Remove model_best_curr")
#     model_list.pop()
# target_model = model_list[-1]

target_model = test_dict["save_folder"]+"model_best_099.pth"

model = torch.load(target_model, map_location=torch.device('cpu'))
print("--->", target_model, " is loaded.")

model = model.to(device)

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['train_list_X'] + data_div['val_list_X'] + data_div['test_list_X']
X_list = sorted(X_list)

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
    x_path = file_path.replace("MR", "CT")
    # x_path = x_path.replace("kspace_2d", "kspace_2d_e80_S2")
    y_path = file_path.replace("kspace_2d_e80_S2", "kspace_2d")
    print(x_path, y_path)
    file_name = os.path.basename(file_path)
    print(" ===> [{:03d}]/[{:03d}]: --->".format(cnt_file+1, total_file), file_name, "<---", end="") #
    x_data = np.load(x_path)
    y_data = np.load(y_path)
    az = x_data.shape[0]
    y_hat = np.zeros(y_data.shape)

    for iz in range(az):

        batch_x = np.zeros((num_vocab, 1, cx**2*2))
        batch_y = np.zeros((num_vocab, 1, cx**2*2))

        batch_x[:, 0, :] = x_data[iz, :, :]
        batch_y[:, 0, :] = y_data[iz, :, :]

        batch_x = torch.from_numpy(batch_x).float().to(device).contiguous()
        batch_y = torch.from_numpy(batch_y).float().to(device).contiguous()
            
        y_hat_iz = model(batch_x, batch_y).detach().cpu().numpy()
        y_hat[iz, :, :] = np.squeeze(y_hat_iz)

    save_name = y_path.replace("kspace_2d_e80_S2", test_dict["new_stage_folder"])
    np.save(save_name, y_hat)
    print(save_name)
