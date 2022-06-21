import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    # "unet",
    # "unet_control_1e1",
    # "unet_control_1e8",
    # "unet_control_1e3",
    # "unet_control_1e3_mse",
    # "unet_control_ob_KL",
    # "unet_control_ob_KL_small",
    # "unet_BNN_KLe5",
    # "unet_BNN_KLe8",
    # "unet_BNN_KLe11",
    # "BNN_KLe6_flip",
    # "unet_dropout10",
    # "unet_dropout25",
    # "unet_dropout50",
    # "unet_dropout75",
    # "do10_MED_1",
    # "do25_MED_1",
    # "do50_MED_1",
    # "do10_MED_5",
    # "do10_MED_15",
    # "do25_MED_5",
    # "do25_MED_15",
    # "do50_MED_5",
    # "do50_MED_15",
    # "Unet_L1",
    "Gau050_MRCT_ZD",
    "Gau050_MRMR_ZD",
    "Gau025_MRCT_ZD",
    "Gau025_MRMR_ZD",
    "Poi100_MRCT_ZD",
    "Poi100_MRMR_ZD",
    "Poi025_MRCT_ZD",
    "Poi025_MRMR_ZD",
    "S&P025_MRCT_ZD",
    "S&P025_MRMR_ZD",
    "S&P050_MRCT_ZD",
    "S&P050_MRMR_ZD",
    "SPK025_MRCT_ZD",
    "SPK025_MRMR_ZD",
    "SPK050_MRCT_ZD",
    "SPK050_MRMR_ZD",
    ]
hub_CT_folder = [
    # "./project_dir/Bayesian_unet_v1/",
    # "./project_dir/Bayesian_unet_v1_control/",
    # "./project_dir/Bayesian_unet_v2_beta_1e8/",
    # "./project_dir/Bayesian_unet_v3_beta_1e3/",
    # "./project_dir/Bayesian_unet_v4_beta_1e3_mse/",
    # "./project_dir/Bayesian_unet_v5_ob_KL/",
    # "./project_dir/Bayesian_unet_v6_ob_KL_small/",
    # "./project_dir/Bayesian_unet_v8_unet_BNN_KLe5/",
    # "./project_dir/Bayesian_unet_v7_unet_BNN_KLe8/",
    # "./project_dir/Bayesian_unet_v10_unet_BNN_KLe11/",
    # "./project_dir/Bayesian_unet_v16_unet_BNN_KLe6_flip/",
    # "./project_dir/Bayesian_unet_v12_unet_drop10/",
    # "./project_dir/Bayesian_unet_v13_unet_drop25/",
    # "./project_dir/Bayesian_unet_v14_unet_drop50/",
    # "./project_dir/Bayesian_unet_v15_unet_drop75/",
    # "./project_dir/Bayesian_MTGD_v1_unet_do10_MTGD5/",
    # "./project_dir/Bayesian_MTGD_v2_unet_do10_MTGD15/",
    # "./project_dir/Bayesian_MTGD_v3_unet_do25_MTGD5/",
    # "./project_dir/Bayesian_MTGD_v4_unet_do25_MTGD15/",
    # "./project_dir/Bayesian_MTGD_v45unet_do50_MTGD5/",
    # "./project_dir/Bayesian_MTGD_v45unet_do50_MTGD15/",
    # "./project_dir/Unet_Monai_Iman_v2/"
    # "./project_dir/Bayesian_HDMGD_v1_Gau050_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v1_Gau050_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v2_Gau025_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v2_Gau025_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v3_Poi100_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v3_Poi100_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v4_Poi025_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v4_Poi025_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v5_S&P025_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v5_S&P025_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v6_S&P050_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v6_S&P050_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v7_SPK025_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v7_SPK025_MRMR/",
    # "./project_dir/Bayesian_HDMGD_v8_SPK050_MRCT/",
    # "./project_dir/Bayesian_HDMGD_v8_SPK050_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v1_Gau050_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v1_Gau050_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v2_Gau025_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v2_Gau025_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v3_Poi100_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v3_Poi100_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v4_Poi025_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v4_Poi025_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v5_S&P025_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v5_S&P025_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v6_S&P050_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v6_S&P050_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v7_SPK025_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v7_SPK025_MRMR/",
    "./project_dir/Bayesian_ZDMGD_v8_SPK050_MRCT/",
    "./project_dir/Bayesian_ZDMGD_v8_SPK050_MRMR/",
    ]


train_loss = []
val_loss = []

for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
    list_train_loss = sorted(glob.glob(CT_folder+"/loss/*train*.npy"))
    list_val_loss = sorted(glob.glob(CT_folder+"/loss/*val*.npy"))
    model_name = hub_CT_name[cnt_CT_folder]
    curr_train_loss = np.zeros((len(list_train_loss), 2))
    curr_val_loss = np.zeros((len(list_val_loss), 2))
    for cnt_epoch, filepath in enumerate(list_train_loss):
        print(filepath)
        data = np.load(filepath)
        if len(data.shape) > 1:
            if data.shape[1]> 1:
                curr_train_loss[cnt_epoch, 0] = np.mean(data[:, 0])
                curr_train_loss[cnt_epoch, 1] = np.mean(data[:, 1])
            else:
                curr_train_loss[cnt_epoch, 0] = np.mean(data)
        else:
            curr_train_loss[cnt_epoch, 0] = np.mean(data)
    for cnt_epoch, filepath in enumerate(list_val_loss):
        print(filepath)
        data = np.load(filepath)
        if len(data.shape) > 1:
            if data.shape[1]> 1:
                curr_val_loss[cnt_epoch, 0] = np.mean(data[:, 0])
                curr_val_loss[cnt_epoch, 1] = np.mean(data[:, 1])
            else:
                curr_val_loss[cnt_epoch, 0] = np.mean(data)
        else:
            curr_val_loss[cnt_epoch, 0] = np.mean(data)
    train_loss.append([model_name, curr_train_loss])
    val_loss.append([model_name, curr_val_loss])

all_loss = [train_loss, val_loss]
save_name = "./metric_bayesian/LOSS_"+"_".join(hub_CT_name)+".npy"
print(save_name)
np.save(save_name, np.asarray(all_loss, dtype=object))

