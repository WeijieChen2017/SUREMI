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
    return np.mean(np.sqrt(np.sum(np.square(x-y))))

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

def std_region(std, ct, tissue):
    if tissue == "air":
        y_mask = filter_data(ct, -2000, -500)
    if tissue == "soft":
        y_mask = filter_data(ct, -500, 250)
    if tissue == "bone":
        y_mask = filter_data(ct, 250, 3000)
    y_mask = y_mask > 0.5
    select_std = std[y_mask]
    return np.mean(select_std)


def filter_data(data, range_min, range_max):
    mask_1 = data < range_max
    mask_2 = data > range_min
    mask_1 = mask_1.astype(int)
    mask_2 = mask_2.astype(int)
    mask = mask_1 * mask_2
    return mask

folder_CT_GT = "./data_dir/Iman_CT/norm/"
hub_CT_name = [
    # "SUnetR_L2",
    "22222",
    "33333",
    "24842",
    "84248",
    ]
hub_CT_folder = [
    # "./project_dir/SwinUNETR_Iman_v4_mse/pred_monai/",
    "./project_dir/MDO_v1_222222222/pred_monai/",
    "./project_dir/MDO_v2_333333333/pred_monai/",
    "./project_dir/MDO_v3_224484422/pred_monai/",
    "./project_dir/MDO_v4_884424488/pred_monai/",
]

hub_metric = ["rmse", "nrmse", "mae", "ssim", "psnr", "acutance", 
              "dice_air", "dice_soft", "dice_bone",
              "std_air", "std_soft", "std_bone",]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(hub_CT_folder[current_model_idx])
time.sleep(1)

cnt_CT_folder = current_model_idx
CT_folder = hub_CT_folder[cnt_CT_folder]

# for cnt_CT_folder, CT_folder in enumerate(hub_CT_folder):
list_std_folder = sorted(glob.glob(CT_folder+"*_std.nii.gz"))
cnt_file_CT = len(list_std_folder)
cnt_metric = len(hub_metric)

table_metric = np.zeros((cnt_file_CT, cnt_metric))

for cnt_CT, path_std in enumerate(list_std_folder):
    print(hub_CT_name[cnt_CT_folder]+" ===> [{:03d}]/[{:03d}]: --->".format(cnt_CT+1, cnt_file_CT), path_std, "<---")
    path_CT = path_std.replace("_std", "")
    filename = os.path.basename(path_CT)
    path_CT_GT = folder_CT_GT+filename
    file_std = nib.load(path_std)
    file_CT = nib.load(path_CT)
    file_CT_GT = nib.load(path_CT_GT)
    data_std = file_std.get_fdata()
    data_CT = file_CT.get_fdata()
    data_CT_GT = file_CT_GT.get_fdata()
    
    data_x = denorm_CT(data_CT)
    data_y = denorm_CT(data_CT_GT)
    
    table_metric[cnt_CT, 0] = mean_squared_error(data_x, data_y)
    table_metric[cnt_CT, 1] = np.sqrt(mean_squared_error(data_x, data_y))
    table_metric[cnt_CT, 2] = mae(data_x, data_y)
    table_metric[cnt_CT, 3] = ssim(data_x, data_y, data_range=4000)
    table_metric[cnt_CT, 4] = psnr(data_x, data_y, data_range=4000)
    table_metric[cnt_CT, 5] = acutance(data_x)
    table_metric[cnt_CT, 6] = dice_coe(data_x, data_y, tissue="air")
    table_metric[cnt_CT, 7] = dice_coe(data_x, data_y, tissue="soft")
    table_metric[cnt_CT, 8] = dice_coe(data_x, data_y, tissue="bone")
    table_metric[cnt_CT, 9] = std_region(data_std, data_y, tissue="air")
    table_metric[cnt_CT, 10] = std_region(data_std, data_y, tissue="soft")
    table_metric[cnt_CT, 11] = std_region(data_std, data_y, tissue="bone")

save_name = "./metric/"+hub_CT_name[cnt_CT_folder]+"_"+"_".join(hub_metric)+".npy"
print(save_name)
np.save(save_name, table_metric)