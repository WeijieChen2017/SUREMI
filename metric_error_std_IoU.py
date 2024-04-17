# here for each case, we compute IoU between error and std, with incremental percentage
import numpy as np
import nibabel as nib
import glob
import os

pred_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*xte.nii.gz"
std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"
ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"

pred_files = sorted(glob.glob(pred_folder))
std_files = sorted(glob.glob(std_folder))
ct_files = sorted(glob.glob(ct_folder))

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
    case_dict_list[case_id] = case_dict


# load the data and compute the case-level std and error
for case_id in case_dict_list.keys():
    case_dict = case_dict_list[case_id]
    pred = nib.load(case_dict["pred"]).get_fdata()
    std = nib.load(case_dict["std"]).get_fdata()
    ct = nib.load(case_dict["ct"]).get_fdata()
    error = np.abs(pred - ct)
    case_dict["error"] = np.mean(error) * 4000
    case_dict["std"] = np.mean(std) * 4000