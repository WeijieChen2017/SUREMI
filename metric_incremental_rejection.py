# here we load all results from the testing dataset, compute the case-level std and error

import numpy as np
import nibabel as nib
import glob
import os

pred_folder_list = [
    # ["project_dir/Unet_Monai_Iman_v2/pred_monai/*zte.nii.gz", "singleU-Net"],
    ["project_dir/Theseus_v2_181_200_rdp1/pred_monai/*xte.nii.gz", "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz", "E2B2D2"],
    ["project_dir/Theseus_v2_47_57_rdp100/pred_monai/*xte.nii.gz", "project_dir/Theseus_v2_47_57_rdp100/pred_monai/*std.nii.gz", "E2B2D2_early"],
    ["project_dir/syn_DLE_1114444_e400_lrn4/full_val_xte/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/full_val_xte/*xte_std.nii.gz", "E1B4D4_full"],
    ["project_dir/syn_DLE_1114444_e400_lrn4/part_val/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/part_val/*xte_std.nii.gz", "E1B4D4_pruned"],
    ["project_dir/syn_DLE_4444111_e400_lrn4/full_val_xte/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/full_val_xte/*xte_std.nii.gz", "E4B4D1_full"],
    ["project_dir/syn_DLE_4444111_e400_lrn4/part_val/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/part_val/*xte_std.nii.gz", "E4B4D1_pruned"],
    ["project_dir/syn_DLE_4444444_e400_lrn4/full_val_xte/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/full_val_xte/*xte_std.nii.gz", "E4B4D4_full"],
    ["project_dir/syn_DLE_4444444_e400_lrn4/part_val/*xte.nii.gz", "project_dir/syn_DLE_1114444_e400_lrn4/part_val/*xte_std.nii.gz", "E4B4D4_pruned"],

]

# std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"

for idx, pred_std_pair in enumerate(pred_folder_list):
    pred_folder = pred_std_pair[0]
    std_folder = pred_std_pair[1]
    save_tag = pred_std_pair[2]
    print(f"Processing {save_tag}...")

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


    # load the data and compute the case-level std and error
    for case_id in case_dict_list.keys():
        case_dict = case_dict_list[case_id]
        pred = nib.load(case_dict["pred"]).get_fdata()
        std = nib.load(case_dict["std"]).get_fdata()
        ct = nib.load(case_dict["ct"]).get_fdata()
        mr = nib.load(case_dict["mr"]).get_fdata()

        mask = mr > np.percentile(mr, 0.05)

        error = np.abs(pred - ct)
        error = error[mask]
        std = std[mask]
        
        case_dict["error"] = np.mean(error) * 4000
        case_dict["std"] = np.mean(std) * 4000
        print(f"case_id {case_id}, error {case_dict['error']}, std {case_dict['std']}")

    save_folder = "results/incremental_rejection/"
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, f"case_dict_list_{save_tag}.npy"), case_dict_list)