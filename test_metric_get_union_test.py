import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix
from skimage.metrics import mean_squared_error

def denorm_CT(data):
    data *= 4000
    data -= 1024
    return data

def rmse(x,y):
    return np.sqrt(np.mean(np.sum(np.square(x-y))))

def nrmse(x,y):
    # return rmse(x,y)
    return 1e6

def mae(x,y):
    return np.mean(np.absolute(x-y))

def acutance(x):
    return np.mean(np.absolute(sobel(data_x)))

def dice_coe(x, y, tissue="air"):
    if tissue == "air":
        x_mask = filter_data(x, -2000, -500)
        y_mask = filter_data(y, -2000, -500)
    if tissue == "soft":
        x_mask = filter_data(x, -500, 250)
        y_mask = filter_data(y, -500, 250)
    if tissue == "bone":
        x_mask = filter_data(x, 250, 3000)
        y_mask = filter_data(y, 250, 3000)
    CM = confusion_matrix(np.ravel(x_mask), np.ravel(y_mask))
    TN, FP, FN, TP = CM.ravel()
    return 2*TP / (2*TP + FN + FP)

def filter_data(data, range_min, range_max):
    mask_1 = data < range_max
    mask_2 = data > range_min
    mask_1 = mask_1.astype(int)
    mask_2 = mask_2.astype(int)
    mask = mask_1 * mask_2
    return mask

folder_MR_GT = "./data_dir/Iman_MR/norm/"
folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    # "Unet_S",
    # "Unet_L",
    # "UnetR_S",
	"UnetR_L",
    # "SwinUnetR_S",
    # "SwinUnetR_L",
    # "SwinIR3d_S",
    # "SwinIR3d_L"
	]
hub_CT_folder = [
    # "./project_dir/Unet_Monai_Iman/pred_monai/",
    # "./project_dir/Unet_Monai_Iman_v1/pred_monai/",
    # "./project_dir/UnetR_Iman_v1/pred_monai/",
    "./project_dir/UnetR_Iman_v2/pred_monai/",
    # "./project_dir/SwinUNETR_Iman_v1/pred_monai/",
    # "./project_dir/SwinUNETR_Iman_v2/pred_monai/",
    # "./project_dir/SwinIR3d_Iman_v1/pred_monai/",
    # "./project_dir/SwinIR3d_Iman_v2/pred_monai/",
]

union_test_file = [
    '02472.nii.gz',
    '04014.nii.gz',
    '05416.nii.gz',
    '05477.nii.gz',
    '06355.nii.gz'
]

for test_filename in union_test_file:
    for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
        cmd = "cp "+CT_folder+test_filename+" ./metric/"+hub_CT_name[cnt_CT_folder]+"_"+test_filename
        print(cmd)
        os.system(cmd)
    cmd = "cp "+folder_CT_GT+test_filename+" ./metric/GT_"+test_filename
    print(cmd)
    os.system(cmd)
    cmd = "cp "+folder_MR_GT+test_filename+" ./metric/MR_"+test_filename
    print(cmd)
    os.system(cmd)
