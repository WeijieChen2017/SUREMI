import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import normalized_root_mse as nrmse
# from sklearn.metrics import mean_squared_error as rmse
# from sklearn.metrics import mean_absolute_error as mae

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

folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = ["SwinUNETR_S", "UnetR_S"]
hub_CT_folder = [
    "./project_dir/SwinUNETR_Iman_v1/pred_monai/",
    "./project_dir/UnetR_Iman_v1/pred_monai/"
]

hub_metric = ["rmse", "nrmse", "mae", "ssim", "psnr", "acutance"]
hub_metric_func = [
    rmse,
    nrmse,
    mae,
    ssim,
    psnr,
    acutance
]

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
        
        data_x = data_CT
        data_y = data_CT_GT
        
        table_metric[cnt_CT, 0] = rmse(data_x, data_y)
        table_metric[cnt_CT, 1] = nrmse(data_x, data_y)
        table_metric[cnt_CT, 2] = mae(data_x, data_y)
        table_metric[cnt_CT, 3] = ssim(data_x, data_y, data_range=4000)
        table_metric[cnt_CT, 4] = psnr(data_x, data_y, data_range=4000)
        table_metric[cnt_CT, 5] = acutance(data_x)
    
    save_name = hub_CT_name[cnt_CT_folder]+"_"+"_".join(hub_metric)+".npy"
    np.save(save_name, table_metric)

