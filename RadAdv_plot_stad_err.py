import os
import numpy as np
import nibabel as nib

model_list = [
    # ["syn_DLE_4444111"],
    # ["syn_DLE_1114444"],
    # ["syn_DLE_4444444"],
    # "Theseus_v2_181_200_rdp1",
    "syn_DLE_4444111_e400_lrn4",
    "syn_DLE_1114444_e400_lrn4",
    "syn_DLE_4444444_e400_lrn4",
    "syn_DLE_2222222_e400_lrn4",
]

folder_name = "part_val"

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
        pred_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/"+filename+"_xte.nii.gz"
        stad_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/"+filename+"_xte_std.nii.gz"

        print(" ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), pred_path, "<---")

        grth = nib.load(grth_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        stad = nib.load(stad_path).get_fdata()
        eror = np.abs(pred-grth)*4000

        
        th_list = np.asarray([50, 100, 150]) / 4000
        n_th = len(th_list)
        mean_seg = np.zeros(n_th+1)
        std_seg = np.zeros(n_th+1)
        counts_seg = np.zeros((n_th+1, 3000))
        seg_map = np.zeros_like(stad)
        for i in range(len(th_list)):
            seg_map[stad > th_list[i]] = i+1

        for i in range(n_th+1):
            mean_seg[i] = np.mean(eror[seg_map == i])
            std_seg[i] = np.std(eror[seg_map == i])

            counts_seg[i, :], bin_edges = np.histogram(
                eror[seg_map == i].flatten(),
                bins=3000, 
                range=(0, 3000)
            )
        
        save_dict = {
            "mean_seg": mean_seg,
            "std_seg": std_seg,
            "counts_seg": counts_seg,
            "bin_edges": bin_edges,
            "dim": stad.shape,
        }

        save_path = "./project_dir/"+test_dict["project_name"]+"/"+folder_name+"/std_HU_eror_"+filename+".npy"
        np.save(save_path, save_dict)