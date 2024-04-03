# here we will load the following files:
# mr data
# ct data
#  diff = mr - ct
# std uncertainty
# Bayesian uncertainty
# EVDL uncertainty
# then we will compare the correlation between the uncertainty and the diff

import os
import glob
import nibabel as nib
import numpy as np

mr_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*xte.nii.gz"
ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"
std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"
bay_folder = "project_dir/Theseus_v2_181_200_rdp1/analysis/*Bayesian.nii.gz"
evdl_folder = "project_dir/Theseus_v2_181_200_rdp1/analysis/_unc_EVDL.nii.gz"

mr_files = sorted(glob.glob(mr_folder))
ct_files = sorted(glob.glob(ct_folder))
std_files = sorted(glob.glob(std_folder))
bay_files = sorted(glob.glob(bay_folder))
evdl_files = sorted(glob.glob(evdl_folder))

def find_filename_with_identifiers(id, filename_list):
    for filename in filename_list:
        if id in filename:
            return filename
    raise ValueError(f"Error finding {id} in the filename list")
    return None

# check the identifier: mr filename is 00008_xte.nii.gz, so we need to extract the 00008
case_id_list = []
for mr_file in mr_files:
    case_id_list.append(mr_file.split("/")[-1].split("_")[0])

case_dict_list = {}
for case_id in case_id_list:
    case_dict = {}
    case_dict["mr"] = find_filename_with_identifiers(case_id, mr_files)
    case_dict["ct"] = find_filename_with_identifiers(case_id, ct_files)
    case_dict["std"] = find_filename_with_identifiers(case_id, std_files)
    case_dict["bay"] = find_filename_with_identifiers(case_id, bay_files)
    case_dict["evdl"] = find_filename_with_identifiers(case_id, evdl_files)
    case_dict_list[case_id] = case_dict


    
