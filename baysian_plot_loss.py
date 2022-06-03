import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    "unet",
    "unet_control_1e1",
    "unet_control_1e8",
    "unet_control_1e3",
    "unet_control_1e3_mse",
    ]
hub_CT_folder = [
    "./project_dir/Bayesian_unet_v1/",
    "./project_dir/Bayesian_unet_v1_control/",
    "./project_dir/Bayesian_unet_v2_beta_1e8/",
    "./project_dir/Bayesian_unet_v3_beta_1e3/",
    "./project_dir/Bayesian_unet_v4_beta_1e3_mse/",
    ]

train_loss = []
val_loss = []

for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
    list_train_loss = sorted(glob.glob(CT_folder+"loss/*train*.npy"))
    list_val_loss = sorted(glob.glob(CT_folder+"loss/*val*.npy"))
    model_name = hub_CT_name[cnt_CT_folder]
    curr_train_loss = np.zeros((len(list_train_loss), 2))
    curr_val_loss = np.zeros((len(list_val_loss), 2))
    for cnt_epoch, filepath in enumerate(list_train_loss):
        print(filepath)
        data = np.load(filepath)
        curr_train_loss[cnt_epoch, 0] = np.mean(data[:, 0])
        curr_train_loss[cnt_epoch, 1] = np.mean(data[:, 1])
    for cnt_epoch, filepath in enumerate(list_val_loss):
        print(filepath)
        data = np.load(filepath)
        curr_val_loss[cnt_epoch, 0] = np.mean(data[:, 0])
        curr_val_loss[cnt_epoch, 1] = np.mean(data[:, 1])
    train_loss.append([model_name, curr_train_loss])
    val_loss.append([model_name, curr_val_loss])

all_loss = [train_loss, val_loss]
save_name = "./metric_bayesian/LOSS_"+"_".join(hub_CT_name)+".npy"
print(save_name)
np.save(save_name, np.asarray(all_loss, dtype=object))

