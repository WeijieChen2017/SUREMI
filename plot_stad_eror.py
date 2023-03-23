import os
import numpy as np
import nibabel as nib

model_list = [
    # ["syn_DLE_4444111"],
    # ["syn_DLE_1114444"],
    # ["syn_DLE_4444444"],
    "Theseus_v2_181_200_rdp1",
    "syn_DLE_4444111",
    "syn_DLE_1114444",
    "syn_DLE_4444444",
]

for model_name in model_list:

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
        pred_path = "./project_dir/"+test_dict["project_name"]+"/pred_monai/"+filename+"_xte.nii.gz"
        stad_path = "./project_dir/"+test_dict["project_name"]+"/pred_monai/"+filename+"_xte_std.nii.gz"

        print(" ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), pred_path, "<---")

        grth = nib.load(grth_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        stad = nib.load(stad_path).get_fdata()*4000
        eror = np.abs(pred-grth)*4000

        eror_flatten = eror.flatten()
        stad_flatten = stad.flatten()

        N_stad, N_eror = 1000, 100
        eror_bins = np.linspace(0, 1000, N_eror+1)
        stad_bins = np.linspace(0, 100, N_stad+1)

        heatmap, _, _ = np.histogram2d(eror_flatten, stad_flatten, bins=[eror_bins, stad_bins])
        save_path = "./project_dir/"+test_dict["project_name"]+"/pred_monai/stat_std_eror_"+filename+".npy"
        np.save(save_path, heatmap)