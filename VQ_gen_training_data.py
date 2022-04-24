import os
import glob
import time
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom

from scipy.cluster import vq

from  sklearn.cluster import MiniBatchKMeans

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["old_modality"] = "norm"
train_dict["new_modality"] = "VQ3d"
train_dict["seed"] = 426
np.random.seed(train_dict["seed"])
train_dict["cube_size"] = 32
train_dict["downsample"] = 8
train_dict["pixel_per_patch"] = train_dict["cube_size"] // train_dict["downsample"]
# train_dict["file_cnt"] = 5
# train_dict["val_ratio"] = 0.3
# train_dict["test_ratio"] = 0.2
cs = train_dict["cube_size"]
ppp = train_dict["pixel_per_patch"]

new_folder_X = train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])
new_folder_Y = train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])

# if not os.path.exists(new_folder_X):
#     os.makedirs(new_folder_X)
# if not os.path.exists(new_folder_Y):
#     os.makedirs(new_folder_Y)

# X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
# Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

# selected_list = np.asarray(X_list)
# np.random.shuffle(selected_list)
# selected_list = list(selected_list)

# val_list = selected_list[:int(len(selected_list)*train_dict["val_ratio"])]
# val_list.sort()
# test_list = selected_list[-int(len(selected_list)*train_dict["test_ratio"]):]
# test_list.sort()
# train_list = list(set(selected_list) - set(val_list) - set(test_list))
# train_list.sort()

# data_division_dict = {
#     "train_list_X" : train_list,
#     "val_list_X" : val_list,
#     "test_list_X" : test_list}
# np.save("./data_dir/VQ3d_8x_data_division.npy", data_division_dict)
# np.save("./data_dir/VQ3d_8x_train_dict.npy", train_dict)

MBK_X = np.load("./data_dir/Iman_MR/VQ3d/MBK_x_cube_8x.npy", allow_pickle=True).item()
MBK_Y = np.load("./data_dir/Iman_CT/VQ3d/MBK_y_cube_8x.npy", allow_pickle=True).item()

std_x = np.load("./data_dir/Iman_MR/VQ3d/std_x_cube_8x.npy", allow_pickle=True)
std_y = np.load("./data_dir/Iman_CT/VQ3d/std_y_cube_8x.npy", allow_pickle=True)

CB_list = np.load("./data_dir/VQ3d_8x_data_division.npy", allow_pickle=True).item()
CB_list = CB_list["train_list_X"]+CB_list["val_list_X"]
CB_list.sort()
for path in CB_list:
    print(path)

onehot_X_array = []
onehot_Y_array = []

# array_x_cube = np.zeros((train_dict["file_cnt"]*384*len(CB_list), ppp**3))
# array_y_cube = np.zeros((train_dict["file_cnt"]*384*len(CB_list), ppp**3))
# cnt_patch = 0

for cnt_file, file_path in enumerate(CB_list):

    total_file = len(CB_list)
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(x_path)
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    ax, ay, az = x_data.shape
    pad_x = int(np.ceil(ax/cs)*cs-ax) // 2
    pad_y = int(np.ceil(ay/cs)*cs-ay) // 2
    pad_z = int(np.ceil(az/cs)*cs-az) // 2
    pad_width = ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z))

    pad_x_data = np.pad(x_data, pad_width)
    pad_y_data = np.pad(y_data, pad_width)

    ds_x = zoom(pad_x_data, 1/train_dict["downsample"])
    ds_y = zoom(pad_y_data, 1/train_dict["downsample"])

    bx, by, bz = ds_x.shape

    onehot_X = np.zeros((bx//ppp, by//ppp, bz//ppp))
    onehot_Y = np.zeros((bx//ppp, by//ppp, bz//ppp))

    for ix in range(bx//ppp):
        for iy in range(by//ppp):
            for iz in range(bz//ppp):
                patch_x = ds_x[ix*ppp:ix*ppp+ppp, iy*ppp:iy*ppp+ppp, iz*ppp:iz*ppp+ppp]
                patch_y = ds_y[ix*ppp:ix*ppp+ppp, iy*ppp:iy*ppp+ppp, iz*ppp:iz*ppp+ppp]
                onehot_X[ix, iy, iz] = MBK_X.predict(np.ravel(patch_x) / std_x)
                onehot_Y[ix, iy, iz] = MBK_X.predict(np.ravel(patch_y) / std_y)

    onehot_X_array.append([x_path, onehot_X])
    onehot_Y_array.append([y_path, onehot_Y])
    
save_name_x = new_folder_X+"onthot_x_cube_8x.npy"
save_name_y = new_folder_Y+"onthot_y_cube_8x.npy"

np.save(save_name_x, array_x_cube)
np.save(save_name_y, array_y_cube)
print(save_name_x, save_name_y)
