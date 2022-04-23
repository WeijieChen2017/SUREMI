import os
import glob
import time
import numpy as np
import nibabel as nib

from scipy.ndimage import zoom

from scipy.cluster import vq

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["old_modality"] = "VQ_8x"
train_dict["new_modality"] = "norm"
train_dict["cube_size"] = 32
train_dict["downsample"] = 8
train_dict["pixel_per_patch"] = train_dict["cube_size"] // train_dict["downsample"]
train_dict["file_cnt"] = 5
cs = train_dict["cube_size"]
ppp = train_dict["pixel_per_patch"]

new_folder_X = train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])
new_folder_Y = train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])

if not os.path.exists(new_folder_X):
    os.makedirs(new_folder_X)
if not os.path.exists(new_folder_X):
    os.makedirs(new_folder_X)

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

np.save("./data_dir/"+train_dict["new_modality"]+"_dict.npy", train_dict)

# array_x_cube = np.zeros((train_dict["file_cnt"]*512, train_dict["cube_size"]**3))
# array_y_cube = np.zeros((train_dict["file_cnt"]*512, train_dict["cube_size"]**3))
# cnt_cube = 0

array_x_patch = np.zeros((train_dict["file_cnt"]*160*64, ppp**2))
array_y_patch = np.zeros((train_dict["file_cnt"]*160*64, ppp**2))
cnt_patch = 0

for cnt_file, file_path in enumerate(X_list[:train_dict["file_cnt"]]):

    total_file = len(X_list)
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
    pad_width = ((pad_x, pad_x), (pad_y, pad_y))

    for iz in range(az):
        img_x = np.pad(x_data[:, :, iz], pad_width)
        img_y = np.pad(y_data[:, :, iz], pad_width)
        
        ds_x = zoom(img_x, 1/train_dict["downsample"])
        ds_y = zoom(img_y, 1/train_dict["downsample"])

        bx, by = ds_x.shape

        for ix in range(bx//ppp):
            for iy in range(by//ppp):
                patch_x = ds_x[ix*ppp:ix*ppp+ppp, iy*ppp:iy*ppp+ppp]
                patch_y = ds_y[ix*ppp:ix*ppp+ppp, iy*ppp:iy*ppp+ppp]
                array_x_patch[cnt_patch, :] = np.ravel(patch_x)
                array_y_patch[cnt_patch, :] = np.ravel(patch_y)
                cnt_patch += 1

    print(az, cnt_patch)


    # ax, ay, az = x_data.shape
    # pad_x = int(np.ceil(ax/cs)*cs-ax) // 2
    # pad_y = int(np.ceil(ay/cs)*cs-ay) // 2
    # pad_z = int(np.ceil(az/cs)*cs-az) // 2
    # pad_width = ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z))

    # pad_x_data = np.pad(x_data, pad_width)
    # pad_y_data = np.pad(y_data, pad_width)

    # print("Pad width: ", pad_width)
    # print("Padded data: ", pad_x_data.shape)
    # px, py, pz = pad_x_data.shape

    # for ix in range(px // cs):
    #     for iy in range(py // cs):
    #         for iz in range(pz // cs):
    #             cube_x = pad_x_data[ix*cs:ix*cs+cs,
    #                                 iy*cs:iy*cs+cs,
    #                                 iz*cs:iz*cs+cs]
    #             cube_y = pad_y_data[ix*cs:ix*cs+cs,
    #                                 iy*cs:iy*cs+cs,
    #                                 iz*cs:iz*cs+cs]

    #             array_x_cube[cnt_cube, :] = np.ravel(cube_x)
    #             array_y_cube[cnt_cube, :] = np.ravel(cube_y)
    #             cnt_cube += 1

whitened_X = vq.whiten(array_x_patch)
whitened_Y = vq.whiten(array_y_patch)

code_book_X, mean_dist_X = vq.kmeans(whitened_X, k_or_guess=100, iter=20)
code_book_Y, mean_dist_Y = vq.kmeans(whitened_Y, k_or_guess=100, iter=20)

print(mean_dist_X, mean_dist_Y)

# save_name_x = train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])+"array_x_cube.npy"
# save_name_y = train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])+"array_y_cube.npy"

# np.save(save_name_x, array_x_cube)
# np.save(save_name_y, array_y_cube)
# print(save_name_x, save_name_y)




