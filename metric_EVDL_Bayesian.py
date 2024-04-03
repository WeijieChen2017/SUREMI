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
evdl_folder = "project_dir/Theseus_v2_181_200_rdp1/analysis/*_unc_EVDL.nii.gz"

mr_files = sorted(glob.glob(mr_folder))
ct_files = sorted(glob.glob(ct_folder))
std_files = sorted(glob.glob(std_folder))
bay_files = sorted(glob.glob(bay_folder))
evdl_files = sorted(glob.glob(evdl_folder))

def find_filename_with_identifiers(id, filename_list):
    for filename in filename_list:
        if id in filename:
            return filename
    raise ValueError(f"Error finding {id} in {filename_list}")
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
    print(f"case_id {case_id}")
    print(f"mr {case_dict['mr']}")
    print(f"ct {case_dict['ct']}")
    print(f"std {case_dict['std']}")
    print(f"bay {case_dict['bay']}")
    print(f"evdl {case_dict['evdl']}")
    print("")

# load the data and compute the correlation

save_folder = "results/Bayesian_EVDL/"
# create folder if not exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
mask_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/"
analysis_folder = "project_dir/Theseus_v2_181_200_rdp1/analysis/"

# create bins
n_div = 2000
std_bins = np.linespace(np.linspace(0, 200, n_div))
bay_bins = np.linespace(np.linspace(0, 0.02, n_div))
evdl_bins = np.linespace(np.linspace(0, 0.2, n_div))
err_bins = np.linespace(np.linspace(0, 1000, n_div))

error_std_corr = np.zeros((n_div, n_div))
error_bay_corr = np.zeros((n_div, n_div))
error_evdl_corr = np.zeros((n_div, n_div))

for case_dict in case_dict_list:
    mr_file = nib.load(case_dict["mr"])
    mr_data = nib.load(case_dict["mr"]).get_fdata()
    ct_data = nib.load(case_dict["ct"]).get_fdata()
    std_data = nib.load(case_dict["std"]).get_fdata()
    bay_data = nib.load(case_dict["bay"]).get_fdata()
    evdl_data = nib.load(case_dict["evdl"]).get_fdata()
    diff = np.abs(mr_data - ct_data) * 4000
    std_data = np.abs(std_data) * 4000
    bay_data = np.abs(bay_data)
    evdl_data = np.abs(evdl_data)

    # 5% th for mr data to get a mask, i.e. 0.05
    mask = mr_data > np.percentile(mr_data, 0.05)
    # save the mask to the mask folder
    mask_filename = os.path.join(mask_folder, case_dict["mr"].split("/")[-1].replace("xte", "mask"))
    mask_file = nib.Nifti1Image(mask, mr_file.affine, mr_file.header)
    nib.save(mask_file, mask_filename)
    print(f"Saved mask to {mask_filename}")

    # apply the mask
    diff = diff[mask]
    std_data = std_data[mask]
    bay_data = bay_data[mask]
    evdl_data = evdl_data[mask]

    # flatten the data
    diff = diff.flatten()
    std_data = std_data.flatten()
    bay_data = bay_data.flatten()
    evdl_data = evdl_data.flatten()

    # enumerate each pixel to put the pair into the corr
    for i in range(len(diff)):
        std_idx = np.digitize(std_data[i], std_bins)
        bay_idx = np.digitize(bay_data[i], bay_bins)
        evdl_idx = np.digitize(evdl_data[i], evdl_bins)
        err_idx = np.digitize(diff[i], err_bins)

        error_std_corr[err_idx, std_idx] += 1
        error_bay_corr[err_idx, bay_idx] += 1
        error_evdl_corr[err_idx, evdl_idx] += 1

    # save this case to the folder
    save_filename = os.path.join(save_folder, case_dict["mr"].split("/")[-1].replace("xte", "corr"))
    data = {
        "error_std_corr": error_std_corr,
        "error_bay_corr": error_bay_corr,
        "error_evdl_corr": error_evdl_corr
    }
    np.save(save_filename, data)
    print(f"Saved correlation to {save_filename}")


    # std_corr = np.corrcoef(diff.flatten(), std_data.flatten())[0, 1]
    # bay_corr = np.corrcoef(diff.flatten(), bay_data.flatten())[0, 1]
    # evdl_corr = np.corrcoef(diff.flatten(), evdl_data.flatten())[0, 1]
    # print(f"case_id {case_id}")
    # print(f"std_corr {std_corr}")
    # print(f"bay_corr {bay_corr}")
    # print(f"evdl_corr {evdl_corr}")
    # print("")

    # save the data
