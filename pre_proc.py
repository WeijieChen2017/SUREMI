import os
import glob
import time
import numpy as np
import nibabel as nib

pre_proc_dict = {}

pre_proc_dict["dir_orig"] = "./data_dir/MR2CT/"
pre_proc_dict["name_orig"] = "CT__MLAC_*_MNI.nii.gz"
pre_proc_dict["dir_syn"] = "./data_dir/norm_MR/"
pre_proc_dict["is_seg"] = False
pre_proc_dict["range_seg"] = [[0, 2000]]
pre_proc_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

# np.save("./log_dir/log_pre_proc_"+pre_proc_dict["time_stamp"]+".npy", )

file_list = sorted(glob.glob(pre_proc_dict["dir_orig"]+pre_proc_dict["name_orig"]))
for file_path  in file_list:
    print("-"*60)
    print(file_path)
    file_nifty = nib.load(file_path)
    file_data = file_nifty.get_fdata()
    print(np.amax(file_data), np.amin(file_data))

