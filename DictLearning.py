import os
import glob
import time
import numpy as np
import nibabel as nib


train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["new_modality"] = "DL_32_1"
train_dict["old_modality"] = "DL_32_1/"
train_dict["cube_size"] = 32
train_dict["file_cnt"] = 5
cs = train_dict["cube_size"]

new_folder_X = train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])
new_folder_Y = train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])

if not os.path.exists(new_folder_X):
    os.makedirs(new_folder_X)
if not os.path.exists(new_folder_X):
    os.makedirs(new_folder_X)

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

np.save("./data_dir/"+train_dict["new_modality"]+"_dict.npy", train_dict)

array_x_cube = np.zeros((train_dict["cube_size"]**3, train_dict["file_cnt"]*512))
array_y_cube = np.zeros((train_dict["cube_size"]**3, train_dict["file_cnt"]*512))
cnt_cube = 0


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
    pad_z = int(np.ceil(az/cs)*cs-az) // 2
    pad_width = ((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z))

    pad_x_data = np.pad(x_data, pad_width)
    pad_y_data = np.pad(y_data, pad_width)

    print("Pad width: ", pad_width)
    print("Padded data: ", pad_x_data.shape)
    px, py, pz = pad_x_data.shape

    for ix in range(px // cs):
        for iy in range(py // cs):
            for iz in range(pz // cs):
                cube_x = pad_x_data[ix*cs:ix*cs+cs,
                                    iy*cs:iy*cs+cs,
                                    iz*cs:iz*cs+cs]
                cube_y = pad_y_data[ix*cs:ix*cs+cs,
                                    iy*cs:iy*cs+cs,
                                    iz*cs:iz*cs+cs]

                array_x_cube[:, cnt_cube] = np.ravel(cube_x)
                array_y_cube[:, cnt_cube] = np.ravel(cube_y)
                cnt_cube += 1
       
save_name_x = train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])+"array_x_cube.npy"
save_name_y = train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])+"array_y_cube.npy"

np.save(save_name_x, array_x_cube)
np.save(save_name_y, array_y_cube)
print(save_name_x, save_name_y)




