import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import confusion_matrix

def denorm_CT(data):
    data *= 4000
    data -= 1000
    return data

def rmse(x,y):
    return np.sqrt(np.mean(np.square(x-y)))

def mae(x,y):
    return np.mean(np.absolute(x-y))

def acutance(x):
    return np.mean(np.absolute(sobel(x)))

def dice_coe(x, y, tissue="air"):
    if tissue == "air":
        x_mask = filter_data(x, -2000, -500)
        y_mask = filter_data(y, -2000, -500)
    if tissue == "soft":
        x_mask = filter_data(x, -500, 500)
        y_mask = filter_data(y, -500, 500)
    if tissue == "bone":
        x_mask = filter_data(x, 500, 3000)
        y_mask = filter_data(y, 500, 3000)
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

def cal_rmse_mae_ssim_psnr_acut_dice(data_x, data_y):
    """
    Calculate the RMSE, MAE, SSIM, PSNR, Acutance, and Dice coefficient of two CT images.
    :param data_x: The first CT image.
    :param data_y: The second CT image.
    :return: The RMSE, MAE, SSIM, PSNR, Acutance, and Dice coefficient of two CT images.
    """

    metirc_rmse = np.sqrt(np.mean(np.square(data_x-data_y)))
    metirc_mae = np.mean(np.absolute(data_x-data_y))
    metirc_ssim = ssim(data_x, data_y, data_range=4000)
    metirc_psnr = psnr(data_x, data_y, data_range=4000)
    metirc_acutance = np.mean(np.absolute(sobel(data_x)))
    metirc_dice_air = dice_coe(data_x, data_y, tissue="air")
    metirc_dice_soft = dice_coe(data_x, data_y, tissue="soft")
    metirc_dice_bone = dice_coe(data_x, data_y, tissue="bone")
    return [metirc_rmse, metirc_mae, metirc_ssim, metirc_psnr, metirc_acutance, metirc_dice_air, metirc_dice_soft, metirc_dice_bone]

def cal_mae(data_x, data_y):
    """
    Calculate the MAE of two CT images.
    :param data_x: The first CT image.
    :param data_y: The second CT image.
    :return: The MAE of two CT images.
    """
    return np.mean(np.absolute(data_x-data_y))