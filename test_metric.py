import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix

def denorm_CT(data):
    data *= 4000
    data -= 1024
    return data

def rmse(x,y):
    return np.sqrt(np.sum(np.square(x-y)))

def nrmse(x,y):
    return np.mean(rmse(x,y))

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

folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
	"SwinUnetR_L"
	]
hub_CT_folder = [
    "./project_dir/SwinUNETR_Iman_v2/pred_monai/"
]

hub_metric = ["rmse", "nrmse", "mae", "ssim", "psnr", "acutance", "dice_air", "dice_soft", "dice_bone"]

for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
    list_CT_folder = sorted(glob.glob(CT_folder+"*.nii.gz"))
    cnt_file_CT = len(list_CT_folder)
    cnt_metric = len(hub_metric)
    
    table_metric = np.zeros((cnt_file_CT, cnt_metric))
    
    for cnt_CT, path_CT in enumerate(list_CT_folder):
        print(hub_CT_name[cnt_CT_folder]+" ===> [{:03d}]/[{:03d}]: --->".format(cnt_CT+1, cnt_file_CT), path_CT, "<---")
        filename = os.path.basename(path_CT)
        path_CT_GT = folder_CT_GT+filename
        file_CT = nib.load(path_CT)
        file_CT_GT = nib.load(path_CT_GT)
        data_CT = file_CT.get_fdata()
        data_CT_GT = file_CT_GT.get_fdata()
        
        data_x = denorm_CT(data_CT)
        data_y = denorm_CT(data_CT_GT)
        
        table_metric[cnt_CT, 0] = rmse(data_x, data_y)
        table_metric[cnt_CT, 1] = nrmse(data_x, data_y)
        table_metric[cnt_CT, 2] = mae(data_x, data_y)
        table_metric[cnt_CT, 3] = ssim(data_x, data_y, data_range=4000)
        table_metric[cnt_CT, 4] = psnr(data_x, data_y, data_range=4000)
        table_metric[cnt_CT, 5] = acutance(data_x)
        table_metric[cnt_CT, 6] = dice_coe(data_x, data_y, tissue="air")
        table_metric[cnt_CT, 7] = dice_coe(data_x, data_y, tissue="soft")
        table_metric[cnt_CT, 8] = dice_coe(data_x, data_y, tissue="bone")
    
    save_name = "./metric/"+hub_CT_name[cnt_CT_folder]+"_"+"_".join(hub_metric)+".npy"
    print(save_name)
    np.save(save_name, table_metric)