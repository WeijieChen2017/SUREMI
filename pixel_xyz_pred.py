import os
import glob
import copy
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import random

def random_pick(some_list,probabilities):
    x = random.uniform(0,1)
    cumulative_probability=0.0
    for item, item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability:
            break
    return item

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "pixel_xyz"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"

train_dict["folder_X"] = "./data_dir/norm_MR/discrete/"
train_dict["folder_Y"] = "./data_dir/norm_CT/discrete/"

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

n_bin = 128

len_x, len_y, len_z = 256, 256, 182

test_MR = train_dict["folder_X"]+"NORM_097.nii.gz"
MR_file = nib.load(test_MR)
MR_data = MR_file.get_fdata()
MR_pred = np.zeros(MR_data.shape)
print(MR_data.shape)

mesh_x = np.asarray(range(n_bin))

for idz in range(len_z):
    zip_name = "./xyz_inference/z{:03d}.zip".format(idz)
    npy_name = "./pixel_xyzx{:03d}_y{:03d}_z{:03d}_pc.npy".format(255, 255, idz)
    unzip_cmd = "unzip "+zip_name
    print(unzip_cmd)
    os.system(unzip_cmd)
    prob_xyz = np.load(npy_name)
    rm_cmd = "rm "+npy_name
    print(rm_cmd)
    os.system(rm_cmd)
    
    for idx in range(len_x):
        for idy in range(len_y):
            MR_value = int(MR_data[idx, idy, idz])
            if MR_value > 127:
                MR_value = 127
            prob_list = prob_xyz[idx, idy, MR_value, :]
            freq_sum = np.sum(prob_list)
            if freq_sum > 0.0:
                prob_list /= freq_sum
                MR_pred[idx, idy, idz] = random_pick(mesh_x, prob_list)

pred_file = nib.Nifti1Image(MR_pred, MR_file.affine, MR_file.header)
pred_name = "CT_pred.nii.gz"
nib.save(pred_file, pred_name)


median_img = copy.deepcopy(MR_pred)

for ix in range(len_x-2):
    for iy in range(len_y-2):
        for iz in range(len_z-2):
            # the outer boundary
            idx = ix+1
            idy = iy+1
            idz = iz+1
#             target = MR_pred[idx, idy, idz]
            neighborhood = np.ravel(MR_pred[idx-1:idx+2, idy-1:idy+2, idz-1:idz+2])
            median_img[idx, idy, idz] = np.median(neighborhood)

pred_file = nib.Nifti1Image(median_img, MR_file.affine, MR_file.header)
pred_name = "CT_pred_median.nii.gz"
nib.save(pred_file, pred_name)