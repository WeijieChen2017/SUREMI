import os
import glob
import time
import numpy as np
import nibabel as nib

def bin_CT(img, n_bins=1024):
    data_max = np.amax(img)
    data_min = np.amin(img)
    data_squeezed = (img-data_min)/(data_max-data_min)
    data_extended = data_squeezed * n_bins
    data_discrete = data_extended // 1
    return int(data_discrete)

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "pixel_correlation"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

for cnt_file, file_path in enumerate(X_list):
     
    file_name = os.path.basename(file_path)
    cube_x_path = file_path
    cube_y_path = train_dict["folder_Y"] + file_name
    print("--->",cube_x_path,"<---")
    cube_x_data = nib.load(cube_x_path).get_fdata()
    cube_y_data = nib.load(cube_y_path).get_fdata()
    len_x, len_y, len_z = cube_x_data.shape

    X_discrete = bin_CT(cube_x_data, n_bins=1024)
    Y_discrete = bin_CT(cube_y_data, n_bins=1024)
    
    pixel_corr = np.zeros((1024, 1024))

    for ix in range(len_x):
    	for iy in range(len_y):
    		for iz in range(len_z):
    			pixel_corr[X_discrete[ix, iy, iz], Y_discrete[ix, iy, iz]] += 1

    np.save(train_dict["save_folder"]+file_name+"_pix_cor.npy")


