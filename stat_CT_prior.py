# Imports
import os
import glob
import time
import numpy as np
import nibabel as nib
import torch

from monai.inferers import sliding_window_inference
from model import UNet_MDO as UNet
from utils import iter_all_order

import os
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference
from utils import iter_all_order


# Configuration dictionary
config = {
    "project_name": "Theseus_v2_181_200_rdp1",
}
def process_data(file_list, config):
    """
    Process a list of data files for the CT prior, save the ratio results.
    Air: (-1000, -500) / 4000
    Soft tissue: (-500, 250) / 4000
    Bone: (250, 3000) / 4000
    the spacing for x, y, z is 1.0, 1.0, 1.0
    so we don't interpolate, we pad the data in z-axis assuming the data is centered

    Parameters:
    - file_list: A list of paths to the input files.
    - config: Dictionary containing configuration and parameters for processing.
    """
    n_file = len(file_list)
    cnt_air = np.zeros((256, 256, 256))
    cnt_soft = np.zeros((256, 256, 256))
    cnt_bone = np.zeros((256, 256, 256))
    for idx, file_path in enumerate(file_list):

        ct_path = file_path.replace("x", "y")
        file_name = os.path.basename(file_path)
        print(f"[{idx+1}]/[{n_file}]: Processing: {file_path}")

        # Load data
        ct_file = nib.load(ct_path)
        ct_data = ct_file.get_fdata()

        # Pad the data
        pad_size = 256 - ct_data.shape[2]
        if pad_size < 0:
            print("The z dimension of the data is larger than 256.")
            continue
        print("--> The shape is padded from", ct_data.shape, "to", (ct_data.shape[0], ct_data.shape[1], 256))
        ct_data = np.pad(ct_data, ((0, 0), (0, 0), (pad_size//2, pad_size-pad_size//2)))
        

        # segmentation
        shifted_ct = ct_data + 1000
        mask_air = shifted_ct < -500 / 4000
        mask_bone = shifted_ct > 250 / 4000
        mask_soft = 1 - mask_air - mask_bone

        # count the segmentation
        cnt_air += np.array(mask_air, dtype=np.int)
        cnt_soft += np.array(mask_soft, dtype=np.int)
        cnt_bone += np.array(mask_bone, dtype=np.int)

    # save the results
    save_folder = config["prior_folder"]
    save_name = "prior_CT.npy"
    prior_air = cnt_air / n_file
    prior_soft = cnt_soft / n_file
    prior_bone = cnt_bone / n_file
    prior_CT = {
        "prior_air": prior_air,
        "prior_soft": prior_soft,
        "prior_bone": prior_bone,
        "n_file": n_file,
    }
    np.save(save_folder+save_name, prior_CT)
    print(f"Prior CT is saved to {save_folder+save_name}")

def main():
    config["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    config["save_folder"] = "./project_dir/"+config["project_name"]+"/"
    config["prior_folder"] = "./project_dir/"+config["project_name"]+"/prior/"
    if not os.path.exists(config["prior_folder"]):
        os.makedirs(config["prior_folder"])

    # Data division
    data_div = np.load(os.path.join(config["save_folder"], "data_division.npy"), allow_pickle=True).item()
    X_list = data_div['train_list_X'] + data_div['val_list_X']
    process_data(X_list, config)

if __name__ == "__main__":
    main()
