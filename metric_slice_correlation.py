# here for each case, we compute IoU between error and std, with incremental percentage
import numpy as np
import nibabel as nib
import glob
import os

def find_filename_with_identifiers(id, filename_list):
    for filename in filename_list:
        if id in filename:
            return filename
    raise ValueError(f"Error finding {id} in {filename_list}")

ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"
mr_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*xte.nii.gz"
pred_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*xte.nii.gz"
std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"

ct_files = sorted(glob.glob(ct_folder))
mr_files = sorted(glob.glob(mr_folder))
pred_files = sorted(glob.glob(pred_folder))
std_files = sorted(glob.glob(std_folder))

# check the identifier: mr filename is 00008_xte.nii.gz, so we need to extract the 00008
case_id_list = []
for pred_file in pred_files:
    case_id = pred_file.split("/")[-1].split("_")[0]
    case_id_list.append(case_id)
    # print(case_id)

# wait for input to continue
# input("Press Enter to continue...")

case_dict_list = {}
for case_id in case_id_list:
    case_dict = {}
    case_dict["pred"] = find_filename_with_identifiers(case_id, pred_files)
    case_dict["std"] = find_filename_with_identifiers(case_id, std_files)
    case_dict["ct"] = find_filename_with_identifiers(case_id, ct_files)
    case_dict["mr"] = find_filename_with_identifiers(case_id, mr_files)
    case_dict_list[case_id] = case_dict

slice_correlation_dict = {}

# load the data and compute the case-level std and error
for case_id in case_id_list:
    case_dict = case_dict_list[case_id]
    pred = nib.load(case_dict["pred"]).get_fdata()
    std = nib.load(case_dict["std"]).get_fdata() * 4000
    ct = nib.load(case_dict["ct"]).get_fdata()
    mr = nib.load(case_dict["mr"]).get_fdata()
    mr_file = nib.load(case_dict["mr"])
    error = np.abs(pred - ct) * 4000

    mr_mask_bool = mr > np.percentile(mr, 0.05)
    mr_mask_int = mr_mask_bool.astype(np.float16)
   
    n_slice = mr.shape[2]
    case_slice_correlation = []
    for i in range(n_slice):
        slice_error = error[:, :, i]
        slice_std = std[:, :, i]
        slice_mr_mask = mr_mask_int[:, :, i]
        slice_correlation = np.corrcoef(slice_error[slice_mr_mask == 1], slice_std[slice_mr_mask == 1])[0, 1]
        case_slice_correlation.append(slice_correlation)
    
    slice_correlation_dict[case_id] = case_slice_correlation
    print(f"case_id {case_id}, mean slice_correlation {np.mean(case_slice_correlation)}, std slice_correlation {np.std(case_slice_correlation)}")

results_folder = "results/slice_correlation"
os.makedirs(results_folder, exist_ok=True)
np.save(os.path.join(results_folder, "slice_correlation_dict.npy"), slice_correlation_dict)
print("Slice correlation saved to", os.path.join(results_folder, "slice_correlation_dict.npy"))