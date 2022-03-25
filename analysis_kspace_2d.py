import os
import glob
import time
import numpy as np
import nibabel as nib


train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

train_dict["folder_X"] = "./data_dir/Iman_MR/kspace_2d/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/kspace_2d/"

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.npy"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.npy"))

train_dict["input_size"] = [256, 256]
ax, ay = train_dict["input_size"]
train_dict["cx"] = 32
cx = train_dict["cx"]

num_vocab = (ax//cx) * (ay//cx)

total_file = len(X_list)
max_z = 200

mag_table = np.zeros((total_file, max_z, 5)) # max, min, mean, std, z
ang_table = np.zeros((total_file, max_z, 5)) # max, min, mean, std, z

for cnt_file, file_path in enumerate(X_list):

    print(file_path)
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(x_path)
    x_data = np.load(x_path)
    y_data = np.load(y_path)

    # book = np.zeros((dz, num_vocab, cx*cx*2))
    dz = x_data.shape[0]
    for data in [x_data, y_data]:
        for iz in range(dz):
            # num_vocab, cx*cx
            real_part = np.squeeze(data[iz, :, :cx*cx])
            imag_part = np.squeeze(data[iz, :, cx*cx:])

            cmplx = np.complex(np.ravel(real_part), np.ravel(imag_part))
            mag = np.abs(cmplx)
            ang = np.angle(cmplx)

            mag_table[cnt_file, iz, 0] = np.amax(mag)
            mag_table[cnt_file, iz, 1] = np.amin(mag)
            mag_table[cnt_file, iz, 2] = np.mean(mag)
            mag_table[cnt_file, iz, 3] = np.std(mag)
            mag_table[cnt_file, iz, 4] = dz

            ang_table[cnt_file, iz, 0] = np.amax(ang)
            ang_table[cnt_file, iz, 1] = np.amin(ang)
            ang_table[cnt_file, iz, 2] = np.mean(ang)
            ang_table[cnt_file, iz, 3] = np.std(ang)
            ang_table[cnt_file, iz, 4] = dz


np.save("mag_table.npy", mag_table)
np.save("ang_table.npy", ang_table)



