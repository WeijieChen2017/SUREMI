import os
import glob
import time
import numpy as np
import nibabel as nib


train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/norm_MR/norm/"
train_dict["folder_Y"] = "./data_dir/norm_CT/norm/"
train_dict["new_modality"] = "kspace"
train_dict["old_modality"] = "norm"
if not os.path.exists(train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])):
    os.makedirs(train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"]))
if not os.path.exists(train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])):
    os.makedirs(train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"]))

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

train_dict["input_size"] = [256, 256, 192]
ax, ay, az = train_dict["input_size"]
train_dict["cx"] = 32
cx = train_dict["cx"]

np.save("./data_dir/"+train_dict["new_modality"]+"_dict.npy", train_dict)

num_vocab = (ax//cx) * (ay//cx) * (az//cx)

for cnt_file, file_path in enumerate(X_list):

    total_file = len(X_list)
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(x_path)
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()

    xy_book = []
    for data in [x_data, y_data]:
        book = np.zeros((num_vocab, cx*cx*cx*2))
        dz = data.shape[2]
        pad_data = np.pad(data, ((0,0),(0,0),((az-dz)//2, (az-dz)//2)), 'constant')
        cnt_cube = 0
        for ix in range(ax//cx):
            for iy in range(ay//cx):
                for iz in range(az//cx):
                    cube = pad_data[ix*cx:ix*cx+cx, iy*cx:iy*cx+cx, iz*cx:iz*cx+cx]
                    k_cube = np.fft.fftshift(np.fft.fftn(cube))
                    book[cnt_cube, :cx*cx*cx] = np.ravel(k_cube).real
                    book[cnt_cube, cx*cx*cx:] = np.ravel(k_cube).imag
                    cnt_cube += 1
        xy_book.append(book)

    x_book = xy_book[0]
    y_book = xy_book[1]

    x_save_name = x_path.replace(train_dict["old_modality"], train_dict["new_modality"])
    y_save_name = y_path.replace(train_dict["old_modality"], train_dict["new_modality"])
    x_save_name = x_save_name.replace("nii.gz", "npy")
    y_save_name = y_save_name.replace("nii.gz", "npy")

    np.save(x_save_name, x_book)
    np.save(y_save_name, y_book)
    print(x_book.shape)




