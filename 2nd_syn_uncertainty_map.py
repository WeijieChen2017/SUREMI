import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

project_list = [
    "syn_DLE_4444444_e400_lrn4",
    "syn_DLE_4444111_e400_lrn4",
    "syn_DLE_1114444_e400_lrn4",
    "syn_DLE_2222222_e400_lrn4",
    ]

root_dir = "./project_dir/"
eval_folder = "/full_val/"

# first load all files ends with "_std.nii.gz"
for project_name in project_list:
    print("Project: ", project_name)
    project_dir = root_dir + project_name + eval_folder
    std_files = sorted(glob.glob(project_dir + "*_std.nii.gz"))
    std_files.sort()
    # print("std files: ", std_files)

    # load all std files and round std into integer
    # count the number of each std value from 0 to 3000 and accumulate
    std_list = []
    std_count = np.zeros(3001)
    for std_file in std_files:
        std_img = nib.load(std_file)
        std_data = std_img.get_fdata()
        std_data = np.round(std_data)
        std_list.extend(std_data.flatten())
        std_count += np.bincount(std_data.flatten().astype(int), minlength=3001)

    # calculate the mean, std, quantiles
    mean = np.mean(std_list)
    std = np.std(std_list)
    quantile_divide = [0.1, 0.2, 0.3, 0.4, 0.5,
                       0.6, 0.7, 0.8, 0.9, 1.0]
    quantiles = np.quantile(std_list, quantile_divide)
    print("mean: ", mean, "std: ", std)
    for i in range(len(quantiles)):
        print("quantile: ", quantile_divide[i], "value: ", quantiles[i])
