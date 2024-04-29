# here we will load a group of predictions and ground truth, and calculate the metrics for them

import os
import glob
import numpy as np
import nibabel as nib

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.ndimage import sobel
from sklearn.metrics import confusion_matrix

# here we will do the following metrics:
# 1. RMSE
# 2. MAE
# 3. SSIM
# 4. PSNR
# 5. Acutance

prediction_folder_list = [
    {"name": "E1B4D4_full", "folder": "syn_DLE_1114444_e400_lrn4/full_val_xte", "filename_affix": "_xte"},
    {"name": "E1B4D4_part", "folder": "syn_DLE_1114444_e400_lrn4/part_val", "filename_affix": "_xte"},
    {"name": "E2B2D2_full", "folder": "syn_DLE_2222222_e400_lrn4/full_val_xte", "filename_affix": "_xte"},
    {"name": "E2B2D2_part", "folder": "syn_DLE_2222222_e400_lrn4/part_val", "filename_affix": "_xte"},
    {"name": "E4B4D1_full", "folder": "syn_DLE_4444111_e400_lrn4/full_val_xte", "filename_affix": "_xte"},
    {"name": "E4B4D1_part", "folder": "syn_DLE_4444111_e400_lrn4/part_val", "filename_affix": "_xte"},
    {"name": "E4B4D4_full", "folder": "syn_DLE_4444444_e400_lrn4/full_val_xte", "filename_affix": "_xte"},
    {"name": "E4B4D4_part", "folder": "syn_DLE_4444444_e400_lrn4/part_val", "filename_affix": "_xte"},
    {"name": "EarlyStop", "folder": "Theseus_v2_47_57_rdp100/pred_monai", "filename_affix": "_xte"},
    {"name": "FullPretrain", "folder": "Theseus_v2_181_200_rdp1/pred_monai", "filename_affix": "_xte"},
    {"name": "SingleUNet", "folder": "Unet_Monai_Iman_v2/pred_monai", "filename_affix": "_zte"},
    {"name": "VoxelDropout25", "folder": "RDO_v1_R00100_D25/pred_monai", "filename_affix": "_xte"},
    {"name": "ChannelDropout25", "folder": "RDO_v2_dim3_R100_D25/pred_monai", "filename_affix": "_xte"},
    {"name": "weightedChannelDropout50", "folder": "Theseus_v3_channelDOw_rdp050/pred_monai", "filename_affix": "_xte"},
    {"name": "weightedChannelDropout100", "folder": "Theseus_v3_channelDOw_rdp100/pred_monai", "filename_affix": "_xte"},
    {"name": "Shuffle50", "folder": "Theseus_v4_shuffle_rdp050/pred_monai", "filename_affix": "_xte"},
    {"name": "Shuffle100", "folder": "Theseus_v4_shuffle_rdp100/pred_monai", "filename_affix": "_xte"},
    {"name": "UNETRv3", "folder": "UnetR_Iman_v3_mse/pred_monai", "filename_affix": ""},
    {"name": "UNETRv4", "folder": "UnetR_Iman_v4_mse/pred_monai", "filename_affix": ""},
]

# here we will do the following metrics:
metric_list = [
    {"name": "rmse", "function": "root_mean_squared_error"},
    {"name": "mae", "function": "mean_absolute_error"},
    {"name": "ssim", "function": "ssim"},
    {"name": "psnr", "function": "psnr"},
    {"name": "acutance", "function": "acutance"},
    {"name": "dice_air", "function": "dice_coe", "tissue": "air"},
    {"name": "dice_soft", "function": "dice_coe", "tissue": "soft"},
    {"name": "dice_bone", "function": "dice_coe", "tissue": "bone"},
]

def root_mean_squared_error(data_x, data_y):
    return np.sqrt(np.mean(np.square(data_x - data_y)))

def mean_absolute_error(data_x, data_y):
    return np.mean(np.abs(data_x - data_y))

def acutance(data_y):
    return np.mean(np.abs(sobel(data_y)))

def filter_data(data, range_min, range_max):
    mask_1 = data < range_max
    mask_2 = data > range_min
    mask_1 = mask_1.astype(int)
    mask_2 = mask_2.astype(int)
    mask = mask_1 * mask_2
    return mask

def dice_coe(data_x, data_y, tissue="air"):
    # here we will calculate the dice coefficient for the given tissue
    # for air: -1000 to -500
    # for soft: -500 to 500
    # for bone: 500 to 3000

    if tissue == "air":
        x_mask = filter_data(data_x, -1000, -500)
        y_mask = filter_data(data_y, -1000, -500)
    if tissue == "soft":
        x_mask = filter_data(data_x, -500, 500)
        y_mask = filter_data(data_y, -500, 500)
    if tissue == "bone":
        x_mask = filter_data(data_x, 500, 3000)
        y_mask = filter_data(data_y, 500, 3000)
    CM = confusion_matrix(np.ravel(x_mask), np.ravel(y_mask))
    TN, FP, FN, TP = CM.ravel()
    dice_coef = 2*TP / (2*TP + FN + FP)

    return dice_coef

def calculate_metrics(data_x, data_y, mask, metric_list):
    # here we will calculate the metrics for the given data

    mask_x = data_x[mask]
    mask_y = data_y[mask]

    metrics = {}
    for metric in metric_list:
        if metric["function"] == "root_mean_squared_error":
            metrics[metric["name"]] = root_mean_squared_error(mask_x, mask_y)
        elif metric["function"] == "mean_absolute_error":
            metrics[metric["name"]] = mean_absolute_error(mask_x, mask_y)
        elif metric["function"] == "ssim":
            metrics[metric["name"]] = ssim(mask_x, mask_y, data_range=4000)
        elif metric["function"] == "psnr":
            metrics[metric["name"]] = psnr(mask_x+1000, mask_y+1000, data_range=4000)
        elif metric["function"] == "acutance":
            metrics[metric["name"]] = acutance(mask_y)
        elif metric["function"] == "dice_coe":
            metrics[metric["name"]] = dice_coe(data_x, data_y)[metric["tissue"]]

    return metrics

mr_folder = "./project_dir/Unet_Monai_Iman_v2/pred_monai/*_xte.nii.gz"
mr_list = sorted(glob.glob(mr_folder))
ct_folder = "./project_dir/Unet_Monai_Iman_v2/pred_monai/*_yte.nii.gz"
ct_list = sorted(glob.glob(ct_folder))
case_id_list = [os.path.basename(x).split("_")[0] for x in mr_list]
results_folder = "./results/synthesis_metrics"
os.makedirs(results_folder, exist_ok=True)
n_cases = len(case_id_list)

for prediction_folder in prediction_folder_list:
    print(f"Processing {prediction_folder['name']}")

    model_metric_dict = {}
    for idx_case, case_id in enumerate(case_id_list):
        mr_path = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"+case_id+"_xte.nii.gz"
        ct_path = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"+case_id+"_yte.nii.gz"
        pred_path = f"./project_dir/{prediction_folder['folder']}/{case_id}{prediction_folder['filename_affix']}.nii.gz"

        mr_img = nib.load(mr_path).get_fdata()
        ct_img = nib.load(ct_path).get_fdata()
        pred_img = nib.load(pred_path).get_fdata()
        ct_img = ct_img * 4000 - 1000
        pred_img = pred_img * 4000 - 1000
        # clip the images
        mr_img = np.clip(mr_img, -1000, 3000)
        ct_img = np.clip(ct_img, -1000, 3000)

        # use 0.05 percentile as the mask threshold
        mr_mask_bool = mr_img > np.percentile(mr_img, 0.05)

        # # apply the mask
        # mask_ct = ct_img[mr_mask_bool]
        # mask_pred = pred_img[mr_mask_bool]

        metrics = calculate_metrics(ct_img, pred_img, mr_mask_bool, metric_list)
        model_metric_dict[case_id] = metrics
        print(f"{prediction_folder['name']} -> [{idx_case+1}/{n_cases}] Processing {case_id}", end="")
        print(f" RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.4f}, Acutance: {metrics['acutance']:.4f} Dice Air: {metrics['dice_air']:.4f}, Dice Soft: {metrics['dice_soft']:.4f}, Dice Bone: {metrics['dice_bone']:.4f}")
    
    # save the metrics
    np.save(f"{results_folder}/{prediction_folder['name']}_metrics.npy", model_metric_dict)
    print(f"Metrics saved to {results_folder}/{prediction_folder['name']}_metrics.npy")
        
