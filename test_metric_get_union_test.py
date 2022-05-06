import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


folder_MR_GT = "./data_dir/Iman_MR/norm/"
folder_CT_GT = "./data_dir/Iman_CT/norm/"

hub_CT_name = [
    # "Unet_S",
    # "Unet_L",
    # "UnetR_S",
    # "UnetR_L",
    # "SwinUnetR_S",
    # "SwinUnetR_L",
    # "SwinIR3d_S",
    # "SwinIR3d_L",
    "MRL_MAE_xyz_654", 
    "MRL_MSE_xyz_666", 
    "MRL_MSE_xyz_654", 
    "MRL_sL1_xyz_666",
    ]

hub_CT_folder = [
    # "./project_dir/Unet_Monai_Iman/",
    # "./project_dir/Unet_Monai_Iman_v1/",
    # "./project_dir/UnetR_Iman_v1/",
    # "./project_dir/UnetR_Iman_v2/",
    # "./project_dir/SwinUNETR_Iman_v1/",
    # "./project_dir/SwinUNETR_Iman_v2/",
    # "./project_dir/SwinIR3d_Iman_v1/",
    # "./project_dir/SwinIR3d_Iman_v2/",
    "./project_dir/MRL_Monai_mae/", 
    "./project_dir/MRL_Monai_mse/", 
    "./project_dir/MRL_Monai_mse_vxyz_64_32_16/", 
    "./project_dir/MRL_Monai_smoothL1/", 
]

union_test_file = [
    '02472.nii.gz',
    '04014.nii.gz',
    '05416.nii.gz',
    '05477.nii.gz',
    '06355.nii.gz'
]


for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
    cmd = "cp "+CT_folder+"dict.npy"+" ./metric/"+hub_CT_name[cnt_CT_folder]+"_dict.npy"
    print(cmd)
    os.system(cmd)



# for test_filename in union_test_file:
#     for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
#         cmd = "cp "+CT_folder+test_filename+" ./metric/"+hub_CT_name[cnt_CT_folder]+"_"+test_filename
#         print(cmd)
#         os.system(cmd)
#     cmd = "cp "+folder_CT_GT+test_filename+" ./metric/GT_"+test_filename
#     print(cmd)
#     os.system(cmd)
#     cmd = "cp "+folder_MR_GT+test_filename+" ./metric/MR_"+test_filename
#     print(cmd)
#     os.system(cmd)
