# Imports
import os
import time
import numpy as np
import nibabel as nib

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

    Three prior maps
    [1] P_class: the prior probability of each class, air, soft tissue, bone
        Compute by counting the number of voxels in each class and divide by the total number of files
    [2] P_x: the prior probability of each voxel value
        Compute by counting the number of voxels divied by the total number of voxels, accumulate the prior_x for all the files
    [3] P_x_class: the prior probability of each voxel value in each class
        Compute by counting the number of voxels with the same value in each class and divide by the total number of voxels in each class

    Parameters:
    - file_list: A list of paths to the input files.
    - config: Dictionary containing configuration and parameters for processing.
    """

    # P_class
    n_file = len(file_list)
    prior_class_air = np.zeros((256, 256, 200))
    prior_class_soft = np.zeros((256, 256, 200))
    prior_class_bone = np.zeros((256, 256, 200))

    # P_x
    prior_x = np.zeros(4000)

    # P_x_class
    prior_x_class_air = np.zeros(4000)
    prior_x_class_soft = np.zeros(4000)
    prior_x_class_bone = np.zeros(4000)

    for idx, file_path in enumerate(file_list):

        ct_path = file_path.replace("x", "y")
        file_name = os.path.basename(file_path)
        print(f"[{idx+1}]/[{n_file}]: Processing: {file_path}")

        # Load data
        ct_file = nib.load(ct_path)
        ct_data = ct_file.get_fdata()
        shifted_ct = ct_data * 4000 - 1000

        # flatten the shifted_ct, clip from 0 to 1 and compute prior_x
        shifted_flat = shifted_ct.flatten()
        shifted_flat = np.clip(shifted_flat, -1000, 3000)
        hist, _ = np.histogram(shifted_flat, bins=4000)
        prior_x += hist
        # prior_x += np.bincount((ct_data_flatten*4000).astype(int), minlength=4000)
        
        # segmentation
        mask_air = shifted_ct < -500
        mask_bone = shifted_ct > 250
        mask_soft = np.logical_and(shifted_ct >= -500, shifted_ct <= 250)

        # P_x_class
        mask_air_flatten = mask_air.flatten()
        mask_soft_flatten = mask_soft.flatten()
        mask_bone_flatten = mask_bone.flatten()
        print(mask_air_flatten.shape, mask_soft_flatten.shape, mask_bone_flatten.shape)
        hist_air, _ = np.histogram(shifted_flat[mask_air_flatten], bins=4000)
        hist_soft, _ = np.histogram(shifted_flat[mask_soft_flatten], bins=4000)
        hist_bone, _ = np.histogram(shifted_flat[mask_bone_flatten], bins=4000)
        print(hist_air.shape, hist_soft.shape, hist_bone.shape)
        hist_air /= np.sum(mask_air).astype(float)
        hist_soft /= np.sum(mask_soft).astype(float)
        hist_bone /= np.sum(mask_bone).astype(float)
        prior_x_class_air += hist_air
        prior_x_class_soft += hist_soft
        prior_x_class_bone += hist_bone

        # Pad the data
        pad_size = 200 - ct_data.shape[2]
        if pad_size < 0:
            print("The z dimension of the data is larger than 200.")
            continue
        print("--> The shape is padded from", ct_data.shape, "to", (ct_data.shape[0], ct_data.shape[1], 200))
        mask_air = np.pad(mask_air, ((0, 0), (0, 0), (pad_size//2, pad_size//2)), 'constant', constant_values=0)
        mask_soft = np.pad(mask_soft, ((0, 0), (0, 0), (pad_size//2, pad_size//2)), 'constant', constant_values=0)
        mask_bone = np.pad(mask_bone, ((0, 0), (0, 0), (pad_size//2, pad_size//2)), 'constant', constant_values=0)

        # count the segmentation
        prior_class_air += mask_air
        prior_class_soft += mask_soft
        prior_class_bone += mask_bone

    # post-process the prior
    prior_class_air /= n_file
    prior_class_soft /= n_file
    prior_class_bone /= n_file

    prior_x /= n_file # take each case as one distribution

    prior_x_class_air /= n_file
    prior_x_class_soft /= n_file
    prior_x_class_bone /= n_file

    # save the prior

    save_folder = config["prior_folder"]
    save_name = "prior_CT.npy"
    prior_CT = {
        "prior_class":{
            "air": prior_class_air,
            "soft": prior_class_soft,
            "bone": prior_class_bone
        },
        "prior_x": prior_x,
        "prior_x_class":{
            "air": prior_x_class_air,
            "soft": prior_x_class_soft,
            "bone": prior_x_class_bone
        },
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
