import os
import glob
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

def bin_CT(img, n_bins=1024):
    data_vector = np.ravel(img)
    data_max = np.amax(data_vector)
    data_min = np.amin(data_vector)
    data_squeezed = (data_vector-data_min)/(data_max-data_min)
    data_extended = data_squeezed * n_bins
    data_discrete = data_extended // 1
#     print(data_discrete.shape)
    return np.asarray(list(data_discrete), dtype=np.int)

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "pixel_correlation"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"

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
    
    for iz in range(len_z):
    
        X_discrete = bin_CT(cube_x_data[:, :, iz], n_bins=n_bin-1)
        Y_discrete = bin_CT(cube_y_data[:, :, iz], n_bins=n_bin-1)
        
        for ix in range(n_bin*n_bin):
            pixel_corr[X_discrete[ix], Y_discrete[ix]] += 1

    
    np.save(train_dict["save_folder"]+file_name[:-7]+"_pix_cor.npy", pixel_corr)
    
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
        
    plt.savefig(train_dict["save_folder"]+file_name[:-7]+"_pix_cor.png")
    plt.close()

