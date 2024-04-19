# load two masks and plot them side by side
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

boundary_list = [10, 15, 20, 25]
data_folder = "results/dice_iou/"
case_list = sorted(glob.glob(f"{data_folder}/*/*mask*.nii.gz"))
model_id_list = []
case_id_list = []
for case in case_list:
    model_id = case.split("/")[2]
    if model_id not in model_id_list:
        model_id_list.append(model_id)
    case_id = case.split("/")[3].split("_")[0]
    if case_id not in case_id_list:
        case_id_list.append(case_id)
print(model_id_list)
print(case_id_list)

idx_case_id = 0
idx_z = 50
for model_id in model_id_list:
    case_id = case_id_list[idx_case_id]
    print(f"Processing {model_id} {case_id}...")

    for boundary in boundary_list:
        # 00008_error_mask_err_0.000_100.000_std_0.000_10.000.nii.gz
        error_filename_low = f"{data_folder}/{model_id}/{case_id}_error_mask_err_0.000_100.000_std_0.000_{boundary}.000.nii.gz"
        # 00008_error_mask_err_100.000_3000.000_std_10.000_3000.000.nii.gz
        error_filename_high = f"{data_folder}/{model_id}/{case_id}_error_mask_err_100.000_3000.000_std_{boundary}.000_3000.000.nii.gz"
        # 00008_std_mask_err_0.000_100.000_std_0.000_10.000.nii.gz
        std_filename_low = f"{data_folder}/{model_id}/{case_id}_std_mask_err_0.000_100.000_std_0.000_{boundary}.000.nii.gz"
        # 00008_std_mask_err_100.000_3000.000_std_10.000_3000.000.nii.gz
        std_filename_high = f"{data_folder}/{model_id}/{case_id}_std_mask_err_100.000_3000.000_std_{boundary}.000_3000.000.nii.gz"

        error_file_low = nib.load(error_filename_low).get_fdata()
        error_file_high = nib.load(error_filename_high).get_fdata()
        std_file_low = nib.load(std_filename_low).get_fdata()
        std_file_high = nib.load(std_filename_high).get_fdata()


        plt.figure(figsize=(10, 10), dpi=100)
        plt.subplot(2, 2, 1)
        img_1 = np.rot90(error_file_low[:, :, idx_z])
        plt.imshow(img_1, cmap="gray")
        plt.title(f"Error < {boundary} HU")
        plt.axis("off")

        plt.subplot(2, 2, 2)
        img_2 = np.rot90(std_file_low[:, :, idx_z])
        plt.imshow(img_2, cmap="gray")
        plt.title(f"Std < {boundary} HU")
        plt.axis("off")

        plt.subplot(2, 2, 3)
        img_3 = np.rot90(error_file_high[:, :, idx_z])
        plt.imshow(img_3, cmap="gray")
        plt.title(f"Error >= {boundary} HU")
        plt.axis("off")

        plt.subplot(2, 2, 4)
        img_4 = np.rot90(std_file_high[:, :, idx_z])
        plt.imshow(img_4, cmap="gray")
        plt.title(f"Std >= {boundary} HU")
        plt.axis("off")

        plt.tight_layout()
        plt_savename = f"{data_folder}/{model_id}_{case_id}_mask_{boundary}_idz{idx_z}.png"
        plt.savefig(plt_savename)
        plt.close()
        print(f"--->Saved {plt_savename}")



