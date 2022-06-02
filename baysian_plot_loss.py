import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    "unet",
    "unet_control",
    ]
hub_CT_folder = [
    "./project_dir/Bayesian_unet_v1/",
    "./project_dir/Bayesian_unet_v1_control/",
    ]

train_loss = []
val_loss = []

for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
    list_train_loss = sorted(glob.glob(CT_folder+"loss/*train*.npy"))
    list_val_loss = sorted(glob.glob(CT_folder+"loss/*val*.npy"))
    model_name = hub_CT_name[cnt_CT_folder]
    curr_train_loss = np.zeros((len(list_train_loss)))
    curr_val_loss = np.zeros((len(list_val_loss)))
    for cnt_epoch, filepath in enumerate(list_train_loss):
        print(filepath)
        data = np.load(filepath)
        curr_train_loss[cnt_epoch] = np.mean(data)
    for cnt_epoch, filepath in enumerate(list_val_loss):
        print(filepath)
        data = np.load(filepath)
        curr_val_loss[cnt_epoch] = np.mean(data)
    train_loss.append([model_name, curr_train_loss])
    val_loss.append([model_name, curr_val_loss])

all_loss = [train_loss, val_loss]
save_name = "./metric_bayesian/LOSS_"+"_".join(hub_CT_name)+".npy"
print(save_name)
np.save(save_name, np.asarray(all_loss, dtype=object))

