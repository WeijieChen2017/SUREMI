import os
import glob
import copy
import time
import numpy as np
import nibabel as nib

# ------------------------------CT------------------------------

# pre_proc_dict = {}

# pre_proc_dict["dir_orig"] = "./data_dir/MR2CT/"
# pre_proc_dict["name_orig"] = "CT__MLAC_*_MNI.nii.gz"
# pre_proc_dict["dir_syn"] = "./data_dir/seg_CT/"
# pre_proc_dict["is_seg"] = True
# pre_proc_dict["attr_seg"] = ["air", "soft_tissue", "bone"]
# pre_proc_dict["range_seg"] = [[-1024, -500], [-500, 500], [500, 3000]]
# pre_proc_dict["note"] = []
# pre_proc_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

# for CT
# file_list = sorted(glob.glob(pre_proc_dict["dir_orig"]+pre_proc_dict["name_orig"]))
# for file_path  in file_list:
#     print("-"*60)
#     print(file_path)
#     file_nifty = nib.load(file_path)
#     file_data = file_nifty.get_fdata()
#     print(np.amin(file_data), np.amax(file_data))

#     pre_proc_dict["note"].append(["There are two ranges, [-1024, 2000+], [0, 3000+]"])
#     if np.amin(file_data) >-1:
#         file_data -= 1024

#     pre_proc_dict["note"].append(["For HU values in CT, [-1024, -500] for air, [-500, 500] for soft tissue, [500, 3000] for bone"])
#     pre_proc_dict["note"].append(["For HU values, we don't want a gap, so soft tissue should be [-200, 200], but we use [-500, 500] instead"])
#     for idx, value_range in enumerate(pre_proc_dict["range_seg"]):
#         value_min = value_range[0]
#         value_max = value_range[1]
#         value_seg = copy.deepcopy(file_data)
#         value_seg[value_seg < value_min] = value_min
#         value_seg[value_seg > value_max] = value_min
#         value_seg = ( value_seg - value_min ) / (value_max - value_min)
#         value_seg[value_seg > 0] = 1

#         save_folder = pre_proc_dict["dir_syn"] + pre_proc_dict["attr_seg"][idx] + "/"
#         if not os.path.exists(save_folder):
#             os.makedirs(save_folder)
#         save_name = "NORM_0"+os.path.basename(file_path)[9:11]+".nii.gz"
#         save_nifty = nib.Nifti1Image(value_seg, file_nifty.affine, file_nifty.header)
#         nib.save(save_nifty, save_folder+save_name)
#         print(save_folder+save_name, " "*4, np.amin(value_seg), " "*4, np.amax(value_seg))

# np.save("./log_dir/log_pre_proc_"+pre_proc_dict["time_stamp"]+".npy", pre_proc_dict)


# ------------------------------regular------------------------------

pre_proc_dict = {}

pre_proc_dict["dir_orig"] = "./data_dir/Iman/paired/MR/"
pre_proc_dict["name_orig"] = "*.nii.gz"
pre_proc_dict["dir_syn"] = "./data_dir/Iman_MR/"
pre_proc_dict["is_seg"] = False
pre_proc_dict["attr_seg"] = ["norm"]
pre_proc_dict["range_seg"] = [[0, 3000]]
pre_proc_dict["note"] = []
pre_proc_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())


file_list = sorted(glob.glob(pre_proc_dict["dir_orig"]+pre_proc_dict["name_orig"]))
for file_path  in file_list:
    print("-"*60)
    print(file_path)
    file_nifty = nib.load(file_path)
    file_data = file_nifty.get_fdata()
    scl_slope = file_nifty.dataobj.slope
    scl_inter = file_nifty.dataobj.inter
    file_data = file_data * scl_slope + scl_inter
    print(np.amin(file_data), np.amax(file_data))

    for idx, value_range in enumerate(pre_proc_dict["range_seg"]):
        value_min = value_range[0]
        value_max = value_range[1]
        value_seg = copy.deepcopy(file_data)
        value_seg = value_seg - (np.amin(value_seg) - value_min)

        value_seg[value_seg < value_min] = value_min
        value_seg[value_seg > value_max] = value_min
        value_seg = ( value_seg - value_min ) / (value_max - value_min)

        save_folder = pre_proc_dict["dir_syn"] + pre_proc_dict["attr_seg"][idx] + "/"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.basename(file_path)
        save_nifty = nib.Nifti1Image(value_seg, file_nifty.affine, file_nifty.header)
        nib.save(save_nifty, save_folder+save_name)
        print(save_folder+save_name, " "*4, np.amin(value_seg), " "*4, np.amax(value_seg))

np.save("./log_dir/log_pre_proc_"+pre_proc_dict["time_stamp"]+".npy", pre_proc_dict)
