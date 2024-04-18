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
        return None

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

# pred_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*xte.nii.gz"
# std_folder = "project_dir/Theseus_v2_181_200_rdp1/pred_monai/*std.nii.gz"

ct_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*yte.nii.gz"
mr_folder = "project_dir/Unet_Monai_Iman_v2/pred_monai/*xte.nii.gz"

ct_files = sorted(glob.glob(ct_folder))
mr_files = sorted(glob.glob(mr_folder))

for idx, pred_std_pair in enumerate(pred_folder_list):
    pred_folder = pred_std_pair[0]
    std_folder = pred_std_pair[1]
    save_tag = pred_std_pair[2]
    print(f"Processing {save_tag}...")

    pred_files = sorted(glob.glob(pred_folder))
    std_files = sorted(glob.glob(std_folder))
    
    # check the identifier: mr filename is 00008_xte.nii.gz, so we need to extract the 00008
    case_id_list = []
    for pred_file in pred_files:
        case_id = pred_file.split("/")[-1].split("_")[0]
        case_id_list.append(case_id)
        print(case_id)

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

    std_ladder = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
    error = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    n_ladder = len(std_ladder)
    case_dict_list["ladder"] = {
        "std": std_ladder,
        "error": error
    }

    # load the data and compute the case-level std and error
    for case_id in case_id_list:
        case_dict = case_dict_list[case_id]
        pred = nib.load(case_dict["pred"]).get_fdata()
        std = nib.load(case_dict["std"]).get_fdata()
        ct = nib.load(case_dict["ct"]).get_fdata()
        mr = nib.load(case_dict["mr"]).get_fdata()
        mr_file = nib.load(case_dict["mr"])
        error = np.abs(pred - ct)

        mask = mr > np.percentile(mr, 0.05)
        # set 1 for the mask, 0 for the rest
        mask = mask.astype(np.float16)


        iou_list = []
        dice_list = []
        for i in range(n_ladder):
            th_error = error[i]
            th_std = std_ladder[i]
            # filter the error and std using the mask and threshold
            error_mask = error < th_error
            std_mask = std < th_std
            error_mask = error_mask.astype(np.float16)
            std_mask = std_mask.astype(np.float16)
            total_error = error_mask*mask
            total_std = std_mask*mask
            # save the error mask and std mask using the mr file header and affine
            error_mask_nii = nib.Nifti1Image(total_error, mr_file.affine, mr_file.header)
            std_mask_nii = nib.Nifti1Image(total_std, mr_file.affine, mr_file.header)
            # create folder to save the masks with the model name after results/IoU_dice/
            save_folder = f"results/dice_iou/{save_tag}/"
            os.makedirs(save_folder, exist_ok=True)
            nib.save(error_mask_nii, os.path.join(save_folder, f"{case_id}_error_mask_err_{th_error}_std_{th_std}.nii.gz"))
            nib.save(std_mask_nii, os.path.join(save_folder, f"{case_id}_std_mask_err_{th_error}_std_{th_std}.nii.gz"))
    print(f"Finished {save_tag}...")