import os
import glob
import time
import numpy as np
import nibabel as nib


train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["new_modality"] = "kspace_2d"
train_dict["old_modality"] = "norm"
train_dict["norm_MR_mag"] = 400
train_dict["norm_CT_mag"] = 700
if not os.path.exists(train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"])):
    os.makedirs(train_dict["folder_X"].replace(train_dict["old_modality"], train_dict["new_modality"]))
if not os.path.exists(train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"])):
    os.makedirs(train_dict["folder_Y"].replace(train_dict["old_modality"], train_dict["new_modality"]))

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

train_dict["input_size"] = [256, 256]
ax, ay = train_dict["input_size"]
train_dict["cx"] = 32
cx = train_dict["cx"]

np.save("./data_dir/"+train_dict["new_modality"]+"_dict.npy", train_dict)

num_vocab = (ax//cx) * (ay//cx)

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
    x_pack = [x_data, train_dict["norm_MR_mag"]]
    y_pack = [y_data, train_dict["norm_CT_mag"]]

    for pack in [y_pack]: #x_pack, 

        data = pack[0]
        norm_cnst = pack[1]

        dz = data.shape[2]
        book = np.zeros((dz, num_vocab, cx*cx*2))
        # pad_data = np.pad(data, ((0,0),(0,0),((az-dz)//2, (az-dz)//2)), 'constant')
        
        for iz in range(dz):

            cnt_patch = 0
            for ix in range(ax//cx):
                for iy in range(ay//cx):
                    patch = data[ix*cx:ix*cx+cx, iy*cx:iy*cx+cx, iz]
                    k_patch = np.fft.fftshift(np.fft.fftn(patch))
                    # k_patch /= norm_cnst
                    # print(patch.shape, k_patch.shape, ix*cx, ix*cx+cx, iy*cx, iy*cx+cx)
                    book[iz, cnt_patch, :cx*cx] = np.ravel(k_patch).real
                    book[iz, cnt_patch, cx*cx:] = np.ravel(k_patch).imag
                    cnt_patch += 1
        xy_book.append(book)

    # x_book = xy_book[0]
    y_book = xy_book[0]

    # x_save_name = x_path.replace(train_dict["old_modality"], train_dict["new_modality"])
    y_save_name = y_path.replace(train_dict["old_modality"], train_dict["new_modality"])
    # x_save_name = x_save_name.replace("nii.gz", "npy")
    y_save_name = y_save_name.replace("nii.gz", "npy")

    # np.save(x_save_name, x_book)
    np.save(y_save_name, y_book)
    # print(x_save_name, x_book.shape, train_dict["norm_MR_mag"])
    print(y_save_name, y_book.shape, train_dict["norm_CT_mag"])




