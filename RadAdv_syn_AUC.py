import os
import numpy as np
import nibabel as nib

model_list = [
    # ["syn_DLE_4444111"],
    # ["syn_DLE_1114444"],
    # ["syn_DLE_4444444"],
    ["Theseus_v2_181_200_rdp1", "pred_monai"],
    ["syn_DLE_4444111_e400_lrn4", "full_val_xte"],
    ["syn_DLE_1114444_e400_lrn4", "full_val_xte"],
    ["syn_DLE_4444444_e400_lrn4", "full_val_xte"],
    ["syn_DLE_2222222_e400_lrn4", "full_val_xte"],

    # ["Theseus_v2_47_57_rdp100", "pred_monai"],
    # ["syn_DLE_4444111_e400_lrn4", "part_val"],
    # ["syn_DLE_1114444_e400_lrn4", "part_val"],
    # ["syn_DLE_4444444_e400_lrn4", "part_val"],
    # ["syn_DLE_2222222_e400_lrn4", "part_val"],
]

for model_pair in model_list:

    model_name = model_pair[0]
    folder_name = model_pair[1]

    test_dict = {}
    test_dict["project_name"] = model_name
    test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
    data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
    file_list = data_div['test_list_X']
    file_list.sort()
    cnt_total_file = len(file_list)

    for cnt_file, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)[:5]
        grth_path = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"+filename+"_yte.nii.gz"
        pred_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/"+filename+"_xte.nii.gz"
        stad_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/"+filename+"_xte_std.nii.gz"

        print(" ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), pred_path, "<---")

        y_data = nib.load(grth_path).get_fdata()
        z_data = nib.load(pred_path).get_fdata()
        std_data = nib.load(stad_path).get_fdata()
        error_data = np.abs(z_data - y_data)

        
        # for each standard deviation, count how many voxels are within that range and compute mean and std in that region

        n_bins = 100
        std_nums = np.histogram(std_data.flatten(), bins=n_bins, range=(0, 200/4000))[0]
        std_bins = np.linspace(0, 200/4000, n_bins+1)
        print(std_bins)

        std_means = np.zeros(n_bins+1)
        std_stds = np.zeros(n_bins+1)
        std_ci = np.zeros(n_bins+1)

        for i in range(n_bins):
            target_data = error_data[np.logical_and(std_data >= std_bins[i], std_data < std_bins[i+1])]
            std_means[i] = np.mean(target_data)
            std_stds[i] = np.std(target_data)
            std_ci[i] = 1.96 * std_stds[i] / np.sqrt(len(target_data))

        print("Done")
        
        save_dict = {
            "std_nums": std_nums,
            "std_means": std_means,
            "std_stds": std_stds,
            "std_ci": std_ci,
        }

        save_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/AUC_stat_"+filename+".npy"
        np.save(save_path, save_dict)