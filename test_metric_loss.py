import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    # "Unet_S",
    # "Unet_L",
    # "UnetR_S",
    # "UnetR_L",
    # "SwinUnetR_S",
    # "SwinUnetR_L",
    "SwinIR3d_S",
    "SwinIR3d_L",
    # "MRL_MAE_xyz_654", 
    "Unet_L1", 
    # "MRL_MSE_xyz_654", 
    "Unet_L2",
    "UR_L1", 
    "UR_L2", 
    "SR_L1", 
    "SR_L2",
    ]
hub_CT_folder = [
    # "./project_dir/Unet_Monai_Iman/",
    # "./project_dir/Unet_Monai_Iman_v1/",
    # "./project_dir/UnetR_Iman_v1/",
    # "./project_dir/UnetR_Iman_v2/",
    # "./project_dir/SwinUNETR_Iman_v1/",
    # "./project_dir/SwinUNETR_Iman_v2/",
    "./project_dir/SwinIR3d_Iman_v1/",
    "./project_dir/SwinIR3d_Iman_v2/",
    # "./project_dir/MRL_Monai_mae/", 
    "./project_dir/MRL_Monai_smoothL1/", 
    # "./project_dir/MRL_Monai_mse_vxyz_64_32_16/", 
    "./project_dir/MRL_Monai_mse/", 
    "./project_dir/UnetR_Iman_v4_mae/", 
    "./project_dir/UnetR_Iman_v3_mse/", 
    "./project_dir/SwinUNETR_Iman_v5_mae/", 
    "./project_dir/SwinUNETR_Iman_v4_mse/", 
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
save_name = "./metric/LOSS_"+"_".join(hub_CT_name)+".npy"
print(save_name)
np.save(save_name, np.asarray(all_loss, dtype=object))

