# here for each case, we compute IoU between error and std, with incremental percentage
import numpy as np
import nibabel as nib
import glob
import os

Gaussian_level = np.asarray([10, 20, 50, 100, 200])/3000
Rician_level = np.asarray([10, 20, 50, 100, 200])/3000
Rayleigh_level = np.asarray([10, 20, 50, 100, 200])/3000
Salt_and_pepper_level = np.asarray([0.01, 0.02, 0.05, 0.1, 0.2])
Radial_sampling_parameters = [(300, 256), (240, 256), (180, 256), (120, 256), (60, 256)]
Spiral_sampling_parameters = [(240, 300), (210, 300), (210, 240), (180, 240), (180, 180)]

def get_corrupted_image(case_id, corruption_type, corruption_level):

    if corruption_type == "Gaussian":
        file_path_median = f"{case_id}_xte_corp_Gaussian_{corruption_level}_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Gaussian_{corruption_level}_std.nii.gz"
    elif corruption_type == "Rician":
        file_path_median = f"{case_id}_xte_corp_Rician_{corruption_level}_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Rician_{corruption_level}_std.nii.gz"
    elif corruption_type == "Rayleigh":
        file_path_median = f"{case_id}_xte_corp_Rayleigh_{corruption_level}_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Rayleigh_{corruption_level}_std.nii.gz"
    elif corruption_type == "Salt_and_pepper":
        file_path_median = f"{case_id}_xte_corp_Salt_and_pepper_{corruption_level}_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Salt_and_pepper_{corruption_level}_std.nii.gz"
    elif corruption_type == "Radial":
        file_path_median = f"{case_id}_xte_corp_Radial_({corruption_level[0]}, {corruption_level[1]})_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Radial_({corruption_level[0]}, {corruption_level[1]})_std.nii.gz"
    elif corruption_type == "Spiral":
        file_path_median = f"{case_id}_xte_corp_Spiral_({corruption_level[0]}, {corruption_level[1]})_median.nii.gz"
        file_path_std = f"{case_id}_xte_corp_Spiral_({corruption_level[0]}, {corruption_level[1]})_std.nii.gz"
    else:
        raise ValueError("Unknown corruption type")
    return file_path_median, file_path_std

    # elif corruption_type == "Rician":
    #     file_path_median = f"./data_corruption/00008_xte_corp_Rician_{corruption_level}_median.nii.gz"
    #     file_path_std = f"./data_corruption/00008_xte_corp_Rician_{corruption_level}_std.nii.gz"
    # elif corruption_type == "Rayleigh":
    #     file_path_median = f"./data_corruption/00008_xte_corp_Rayleigh_{corruption_level}_median.nii.gz"
    #     file_path_std = f"./data_corruption/00008_xte_corp_Rayleigh_{corruption_level}_std.nii.gz"
    # elif corruption_type == "Salt_and_pepper":
    #     file_path_median = f"./data_corruption/00008_xte_corp_Salt_and_pepper_{corruption_level}_median.nii.gz"
    #     file_path_std = f"./data_corruption/00008_xte_corp_Salt_and_pepper_{corruption_level}_std.nii.gz"
    # elif corruption_type == "Radial":
    #     file_path_median = f"./data_corruption/00008_xte_corp_Radial_({corruption_level[0]}, {corruption_level[1]})_median.nii.gz"
    #     file_path_std = f"./data_corruption/00008_xte_corp_Radial_({corruption_level[0]}, {corruption_level[1]})_std.nii.gz"
    # elif corruption_type == "Spiral":
    #     file_path_median = f"./data_corruption/00008_xte_corp_Spiral_({corruption_level[0]}, {corruption_level[1]})_median.nii.gz"
    #     file_path_std = f"./data_corruption/00008_xte_corp_Spiral_({corruption_level[0]}, {corruption_level[1]})_std.nii.gz"
    # else:
    #     raise ValueError("Unknown corruption type")
    # return file_path_median, file_path_std

def find_filename_with_identifiers(id, filename_list):
        for filename in filename_list:
            if id in filename:
                return filename
        raise ValueError(f"Error finding {id} in {filename_list}")
        return None

pred_folder = "project_dir/Theseus_v2_181_200_rdp1/corrpution/"
ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"
mr_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*xte.nii.gz"

ct_files = sorted(glob.glob(ct_folder))
mr_files = sorted(glob.glob(mr_folder))

# check the identifier: mr filename is 00008_xte.nii.gz, so we need to extract the 00008
case_id_list = []
for pred_file in mr_files:
    case_id = pred_file.split("/")[-1].split("_")[0]
    case_id_list.append(case_id)
    # print(case_id)
    print(case_id)

case_dict_list = {}
for case_id in case_id_list:
    case_dict = {}
    case_dict["ct"] = find_filename_with_identifiers(case_id, ct_files)
    case_dict["mr"] = find_filename_with_identifiers(case_id, mr_files)
    case_dict_list[case_id] = case_dict 

method_list = ["Gaussian", "Rician", "Rayleigh", "Salt_and_pepper", "Radial", "Spiral"]
level_list = [
    Gaussian_level,
    Rician_level,
    Rayleigh_level,
    Salt_and_pepper_level,
    Radial_sampling_parameters,
    Spiral_sampling_parameters,
]
title_list = [
    "Gaussian noise std:",
    "Rician noise std:",
    "Rayleigh noise std:",
    "Salt&Pepper:",
    "Radial ",
    "Spiral ",
]

# L1 dict is the top level dictionary
L1_dict = {}
# L2 dict is the second level dictionary for each case
# L2_dict_case = {}
# L3 dict is the third level dictionary for each method
# L3_dict_method = {}
# L4 dict is the fourth level dictionary for each level
# L4_dict_level = {}

for case_id in case_id_list:

    print("Processing", case_id)
    mr_path = case_dict_list[case_id]["mr"]
    mr_data = nib.load(mr_path).get_fdata()
    mr_mask_bool = mr_data > np.percentile(mr_data, 0.05)

    ct_path = case_dict_list[case_id]["ct"]
    ct_data = nib.load(ct_path).get_fdata()

    L2_dict_case = {}
    for idx_method, method in enumerate(method_list):

        print(f"[{case_id}] Processing {method}")
        L3_dict_method = {}
        for idx_level, level in enumerate(level_list[idx_method]):

            display_level = level
            # if display_level is a tuple, then we need to convert it to a string
            # if display_level is a float, then we only use 4 decimal points
            if type(display_level) == tuple:
                display_level = f"({display_level[0]}, {display_level[1]})"
            else:
                display_level = f"{display_level:.4f}"

            print(f"[{case_id}] Processing {method} level {display_level}")
            corruption_level = level
            corruption_type = method
            file_path_median, file_path_std = get_corrupted_image(case_id, corruption_type, corruption_level)
            median_data = nib.load(pred_folder+file_path_median).get_fdata()
            std_data = nib.load(pred_folder+file_path_std).get_fdata() * 4000
            error_data = np.abs(median_data - ct_data) * 4000

            # apply mr_mask_bool to the error and std then compute the mean
            error_data_masked = error_data[mr_mask_bool]
            std_data_masked = std_data[mr_mask_bool]
            L4_dict_level = {
                "error_mean": np.mean(error_data_masked),
                "error_std": np.std(error_data_masked),
                "unc_mean": np.mean(std_data_masked),
                "unc_std": np.std(std_data_masked),
            }
            L3_dict_method[display_level] = L4_dict_level
            print(f"[{case_id}] {method} level {display_level} error mean {L4_dict_level['error_mean']:.4f} std {L4_dict_level['error_std']:.4f} unc mean {L4_dict_level['unc_mean']:.4f} std {L4_dict_level['unc_std']:.4f}")
        L2_dict_case[str(method)] = L3_dict_method
    L1_dict[case_id] = L2_dict_case

result_folder = "results/metric_corruption/"
os.makedirs(result_folder, exist_ok=True)
np.save(os.path.join(result_folder, "corruption_metric.npy"), L1_dict)
print("Saved corruption_metric.npy")