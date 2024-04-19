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

    # std_ladder = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
    # error_ladder = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600]
    # err_ladder_qth = [0, 33.3, 66.6, 100]
    # std_ladder_qth = [0, 33.3, 66.6, 100]
    # err_ladder_qth = [0, 25, 50, 75, 100]
    # std_ladder_qth = [0, 25, 50, 75, 100]
    err_ladder = [0, 100, 3000]
    std_ladder = [0, 25, 3000]
    n_ladder = len(err_ladder)
    # case_dict_list["ladder"] = {
    #     "std": std_ladder,
    #     "error": error_ladder
    # }

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
        case_dict["error"] = np.mean(error)
        case_dict["std"] = np.mean(std)

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
        os.makedirs(os.path.join(f"results/dice_iou/{save_tag}/"), exist_ok=True)
        for idx_th in range(n_ladder - 1):
            
            err_th_low = err_ladder[idx_th]
            err_th_high = err_ladder[idx_th + 1]
            std_th_low = std_ladder[idx_th]
            std_th_high = std_ladder[idx_th + 1]

            # err_th_low = np.percentile(error[mr_mask_bool], err_ladder_qth[idx_th])
            # err_th_high = np.percentile(error[mr_mask_bool], err_ladder_qth[idx_th + 1])
            # std_th_low = np.percentile(std[mr_mask_bool], std_ladder_qth[idx_th])
            # std_th_high = np.percentile(std[mr_mask_bool], std_ladder_qth[idx_th + 1])

            error_mask_bool = np.logical_and(error > err_th_low, error <= err_th_high)
            std_mask_bool = np.logical_and(std > std_th_low, std <= std_th_high)

            cross_err_mr_mask = np.logical_and(mr_mask_bool, error_mask_bool)
            cross_std_mr_mask = np.logical_and(mr_mask_bool, std_mask_bool)

            cross_err_mr_mask_int = cross_err_mr_mask.astype(np.float16)
            cross_std_mr_mask_int = cross_std_mr_mask.astype(np.float16)
            cross_err_mr_mask_file = nib.Nifti1Image(cross_err_mr_mask_int, mr_file.affine, mr_file.header)
            cross_std_mr_mask_file = nib.Nifti1Image(cross_std_mr_mask_int, mr_file.affine, mr_file.header)
            # savename should be 3 decimal places
            cross_err_mr_mask_savename = os.path.join(f"results/dice_iou/{save_tag}/", f"{case_id}_error_mask_err_{err_th_low:.3f}_{err_th_high:.3f}_std_{std_th_low:.3f}_{std_th_high:.3f}.nii.gz")
            cross_std_mr_mask_savename = os.path.join(f"results/dice_iou/{save_tag}/", f"{case_id}_std_mask_err_{err_th_low:.3f}_{err_th_high:.3f}_std_{std_th_low:.3f}_{std_th_high:.3f}.nii.gz")
            nib.save(cross_err_mr_mask_file, cross_err_mr_mask_savename)
            nib.save(cross_std_mr_mask_file, cross_std_mr_mask_savename)

            mask_1 = cross_err_mr_mask
            mask_2 = cross_std_mr_mask

            intersection = np.logical_and(mask_1, mask_2)
            union = np.logical_or(mask_1, mask_2)
            iou = np.sum(intersection) / np.sum(union)
            dice = 2 * np.sum(intersection) / (np.sum(mask_1) + np.sum(mask_2))
            print(f"IoU = {iou:.3f}, Dice = {dice:.3f} for std from {std_th_low:.3f} to {std_th_high:.3f} and error from {err_th_low:.3f} to {err_th_high:.3f}")

            iou_list.append(iou)
            dice_list.append(dice)
        
        case_dict["iou"] = iou_list
        case_dict["dice"] = dice_list
        print(f"case_id {case_id}, error {case_dict['error']:.3f}, std {case_dict['std']:.3f}, iou {np.mean(iou_list):.3f}, dice {np.mean(dice_list):.3f}")

    save_folder = "results/dice_iou/"
    os.makedirs(save_folder, exist_ok=True)
    np.save(os.path.join(save_folder, f"case_dict_list_ladders_{save_tag}.npy"), case_dict_list)
    print("Saved to", os.path.join(save_folder, f"case_dict_list_ladders_{save_tag}.npy"))