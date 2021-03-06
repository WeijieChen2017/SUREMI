import os
import glob
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from externel import seaborn as sns

def bin_CT(img, n_bins=1024):
    data_vector = np.ravel(img)
    data_max = np.amax(data_vector)
    data_min = np.amin(data_vector)
    # print(data_max, data_min)
    data_squeezed = (data_vector-data_min)/(data_max-data_min)
    data_extended = data_squeezed * n_bins
    data_discrete = data_extended // 1
#     print(data_discrete.shape)
    return np.asarray(list(data_discrete), dtype=np.int64)

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "pixel_correlation"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT_2/regular/"

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

n_bin = 128

for cnt_file, file_path in enumerate(X_list):
     
    file_name = os.path.basename(file_path)
    cube_x_path = file_path
    cube_y_path = train_dict["folder_Y"] + file_name
    print("--->",cube_x_path,"<---")
    cube_x_data = nib.load(cube_x_path).get_fdata()
    cube_y_data = nib.load(cube_y_path).get_fdata()
    len_x, len_y, len_z = cube_x_data.shape
    pixel_corr = np.zeros((n_bin, n_bin))
    
    X_discrete = bin_CT(cube_x_data, n_bins=n_bin-1)
    Y_discrete = bin_CT(cube_y_data, n_bins=n_bin-1)
        
    for ix in range(len(X_discrete)):
        pixel_corr[X_discrete[ix], Y_discrete[ix]] += 1
    
    for ix in range(n_bin):
        temp_sum = np.sum(pixel_corr[ix, :])
        # print(np.amax(pixel_corr[ix, :]), end="")
        if not temp_sum == 0.0:
            pixel_corr[ix, :] = pixel_corr[ix, :] / np.sum(pixel_corr[ix, :])
        # print(np.amax(pixel_corr[ix, :]))
            
    loc_x = np.zeros((n_bin)*(n_bin))
    loc_y = np.zeros((n_bin)*(n_bin))
    pc_ft = np.zeros((n_bin)*(n_bin))
    for idx in range(n_bin):
        for idy in range(n_bin):
            flatten = idx*n_bin + idy
            loc_x[flatten] = idx
            loc_y[flatten] = idy
            pc_ft[flatten] = pixel_corr[idx, idy]

    corr_mat = pd.DataFrame({"X":loc_x, "Y":loc_y, "counts":pc_ft})

    plt.figure(figsize=(12, 12), dpi=1200)
    g = sns.relplot(
        data=corr_mat,
        x="X", y="Y", hue="counts",
        palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8),
    )

    # Tweak the figure to finalize
    g.set(xlabel="MR", ylabel="CT", aspect="equal")
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")
        
    np.save(train_dict["save_folder"]+file_name[:-7]+"_pix_cor.npy", pixel_corr)
    plt.savefig(train_dict["save_folder"]+file_name[:-7]+"_pix_cor.png")
    plt.close('all')

