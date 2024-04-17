# here for each case, we compute IoU between error and std, with incremental percentage
import numpy as np
import nibabel as nib
import glob
import os

pred_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*xte.nii.gz"
std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"
ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"
mr_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*xte.nii.gz"

pred_files = sorted(glob.glob(pred_folder))
std_files = sorted(glob.glob(std_folder))
ct_files = sorted(glob.glob(ct_folder))
mr_files = sorted(glob.glob(mr_folder))

def find_filename_with_identifiers(id, filename_list):
    for filename in filename_list:
        if id in filename:
            return filename
    raise ValueError(f"Error finding {id} in {filename_list}")
    return None

# check the identifier: mr filename is 00008_xte.nii.gz, so we need to extract the 00008
case_id_list = []
for pred_file in pred_files:
    case_id_list.append(pred_file.split("/")[-1].split("_")[0])

case_dict_list = {}
for case_id in case_id_list:
    case_dict = {}
    case_dict["pred"] = find_filename_with_identifiers(case_id, pred_files)
    case_dict["std"] = find_filename_with_identifiers(case_id, std_files)
    case_dict["ct"] = find_filename_with_identifiers(case_id, ct_files)
    case_dict["mr"] = find_filename_with_identifiers(case_id, mr_files)
    case_dict_list[case_id] = case_dict

std_ladder = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
error = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
n_ladder = len(std_ladder)
case_dict_list["ladder"] = {
    "std": std_ladder,
    "error": error
}

# load the data and compute the case-level std and error
for case_id in case_dict_list.keys():
    case_dict = case_dict_list[case_id]
    pred = nib.load(case_dict["pred"]).get_fdata()
    std = nib.load(case_dict["std"]).get_fdata()
    ct = nib.load(case_dict["ct"]).get_fdata()
    mr = nib.load(case_dict["mr"]).get_fdata()
    error = np.abs(pred - ct)

    mask = mr > np.percentile(mr, 0.05)
    error = error[mask]
    std = std[mask]
    case_dict["error"] = np.mean(error) * 4000
    case_dict["std"] = np.mean(std) * 4000

    # iou_list = []
    # dice_list = []
    # # enumerate i from 1% to 100%
    # for i in range(1, 101):
    #     th_error = np.percentile(error, i)
    #     th_std = np.percentile(std, i)
    #     error_mask = error < th_error
    #     std_mask = std < th_std
    #     # print(f"The {i}th percentile of error is {th_error}, std is {th_std}, error_mask {np.sum(error_mask)}, std_mask {np.sum(std_mask)}")
    #     intersection = np.sum(error_mask & std_mask)
    #     union = np.sum(error_mask | std_mask)
    #     iou = intersection / union
    #     dice = 2 * intersection / (np.sum(error_mask) + np.sum(std_mask))
    #     iou_list.append(iou)
    #     dice_list.append(dice)

    iou_list = []
    dice_list = []
    for i in range(n_ladder):
        th_error = error[i]
        th_std = std_ladder[i]
        error_mask = error < th_error
        std_mask = std < th_std
        intersection = np.sum(error_mask & std_mask)
        union = np.sum(error_mask | std_mask)
        iou = intersection / union
        dice = 2 * intersection / (np.sum(error_mask) + np.sum(std_mask))
        iou_list.append(iou)
        dice_list.append(dice)
    
    case_dict["iou"] = iou_list
    case_dict["dice"] = dice_list
    print(f"case_id {case_id}, error {case_dict['error']}, std {case_dict['std']}, iou {np.mean(iou_list)}, dice {np.mean(dice_list)}")

save_folder = "results/dice_iou/"
os.makedirs(save_folder, exist_ok=True)
np.save(os.path.join(save_folder, "case_dict_list_ladders.npy"), case_dict_list)
print("Saved to", os.path.join(save_folder, "case_dict_list.npy"))