import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.metrics.pairwise import paired_distances

hub_CT_savename = [
    # "22222",
    # "33333",
    # "24842",
    # "84248",
    # "e200_rdp_000",
    # "e200_rdp_020",
    # "e200_rdp_040",
    # "e200_rdp_060",
    # "e200_rdp_080",
    # "e200_rdp_100",
    # "e50_rdp_000",
    # "e50_rdp_020",
    # "e50_rdp_040",
    # "e50_rdp_060",
    # "e50_rdp_080",
    # "e50_rdp_100",
    "R001_D50",
    "R010_D50",
    "R100_D25",
    "R100_D50",
    "R100_D75",
    # "cDO_r050",
    # "cDO_r100",
    # "cDO_w050",
    # "cDO_w100",
    "base_unet",
    ]
hub_CT_folder = [
    # "MDO_v1_222222222",
    # "MDO_v2_333333333",
    # "MDO_v3_224484422",
    # "MDO_v4_884424488",
    # "Theseus_v2_181_200_rdp0",
    # "Theseus_v2_181_200_rdp1",
    # "Theseus_v2_181_200_rdp020",
    # "Theseus_v2_181_200_rdp040",
    # "Theseus_v2_181_200_rdp060",
    # "Theseus_v2_181_200_rdp080",
    # "Theseus_v2_47_57_rdp000",
    # "Theseus_v2_47_57_rdp020",
    # "Theseus_v2_47_57_rdp040",
    # "Theseus_v2_47_57_rdp060",
    # "Theseus_v2_47_57_rdp080",
    # "Theseus_v2_47_57_rdp100",
    "RDO_v1_R00001_D50",
    "RDO_v1_R00010_D50",
    "RDO_v1_R00100_D25",
    "RDO_v1_R00100_D50",
    "RDO_v1_R00100_D75",
    # "Theseus_v3_channelDO_rdp050",
    # "Theseus_v3_channelDO_rdp100",
    # "Theseus_v3_channelDOw_rdp050",
    # "Theseus_v3_channelDOw_rdp100",
    "Unet_Monai_Iman_v2",
]

save_list = []

for idx_model, model_name in enumerate(hub_CT_folder):

    model_folder = "./project_dir/"+model_name+"/model_best_*.pth"
    model_list = sorted(glob.glob(model_folder))
    if "curr" in model_list[-1]:
        print("Remove model_best_curr")
        model_list.pop()
    target_model = model_list[-1]
    model = torch.load(target_model, map_location=torch.device('cpu'))
    print("--------------->", target_model, " is loaded.")

    state_dict = model.state_dict()
    model_state_keys = model.state_dict().keys()
    save_list_sub_sub = []

    for model_key in model_state_keys:
        if model_key.split(".")[-2] == "conv" and model_key.split(".")[-1] == "weight":
            print(model_key)
            sdk = state_dict[model_key].numpy()
            n_kernel = sdk.shape[0]
            n_elem = sdk.shape[1]*sdk.shape[2]*sdk.shape[3]*sdk.shape[4]
            kernels = np.zeros((n_kernel, n_elem))
            for i in range(n_kernel):
                kernels[i, :] = sdk[i, :, :, :, :].flatten()
            
            dist_list_euc = []
            dist_list_cos = []
            for i in range(0, n_kernel):
                for j in range(0, i):
                    dist_list_euc.append(paired_distances([kernels[i, :]], [kernels[j, :]], metric="euclidean"))
                    dist_list_cos.append(paired_distances([kernels[i, :]], [kernels[j, :]], metric="cosine"))
            mean_euc = np.mean(np.asarray(dist_list_euc))
            std_euc = np.std(np.asarray(dist_list_euc))
            mean_cos = np.mean(np.asarray(dist_list_cos))
            std_cos = np.std(np.asarray(dist_list_cos))
            save_list_sub_sub.append([model_key, mean_euc, std_euc, mean_cos, std_cos])

    save_list.append([hub_CT_savename[idx_model], model_folder, save_list_sub_sub])

save_name = "./metric/kernel_consistency.npy"
print(save_name)
np.save(save_name, save_list, dtype=object)