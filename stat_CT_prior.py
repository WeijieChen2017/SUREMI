# Imports
import os
import time
import numpy as np
import nibabel as nib

# Configuration dictionary
config = {
    "project_name": "Theseus_v2_181_200_rdp1",
    "air_soft_midpoint": 500,
    "soft_bone_midpoint": 1500
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

    # P_x_class, prob is strictly related to x, so we don't need to do it respectively
    prior_x_class = np.zeros(4000)
    # prior_x_class_air = np.zeros(4000)
    # prior_x_class_soft = np.zeros(4000)
    # prior_x_class_bone = np.zeros(4000)

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
        
        # P_x_class, by normalizng three ranges
        # # normalize from -1000 to -500, with index from 0 to 499
        # prior_x_class[0:500]  =  prior_x_class[0:500] + hist[0:500] / np.sum(hist[0:500])
        # # normalize from -500 to 250, with index from 500 to 1250
        # prior_x_class[500:1250]  =  prior_x_class[500:1250] + hist[500:1250] / np.sum(hist[500:1250])
        # # normalize from 250 to 3000, with index from 1250 to 4000
        # prior_x_class[1250:4000]  =  prior_x_class[1250:4000] + hist[1250:4000] / np.sum(hist[1250:4000])

        # normalize from -1000 to -500, with index from 0 to 500
        prior_x_class[:config["air_soft_midpoint"]] = prior_x_class[:config["air_soft_midpoint"]] + hist[:config["air_soft_midpoint"]] / np.sum(hist[:config["air_soft_midpoint"]])
        # normalize from -300 to 500, with index from 700 to 1500
        prior_x_class[config["air_soft_midpoint"]:config["soft_bone_midpoint"]] = prior_x_class[config["air_soft_midpoint"]:config["soft_bone_midpoint"]] + hist[config["air_soft_midpoint"]:config["soft_bone_midpoint"]] / np.sum(hist[config["air_soft_midpoint"]:config["soft_bone_midpoint"]])
        # normalize from 500 to 3000, with index from 1500 to 4000
        prior_x_class[config["soft_bone_midpoint"]:] = prior_x_class[config["soft_bone_midpoint"]:] + hist[config["soft_bone_midpoint"]:] / np.sum(hist[config["soft_bone_midpoint"]:])

        # segmentation
        # mask_air = shifted_ct < -500
        # mask_bone = shifted_ct > 250
        # mask_soft = np.logical_and(shifted_ct >= -500, shifted_ct <= 250)

        mask_air = shifted_ct < config["air_soft_midpoint"] - 1000
        mask_bone = shifted_ct > config["soft_bone_midpoint"] - 1000
        mask_soft = np.logical_and(shifted_ct >= config["air_soft_midpoint"] - 1000, shifted_ct <= config["soft_bone_midpoint"] - 1000)

        # P_x_class
        # mask_air_flatten = mask_air.flatten()
        # mask_soft_flatten = mask_soft.flatten()
        # mask_bone_flatten = mask_bone.flatten()
        # hist_air, _ = np.histogram(shifted_flat[mask_air_flatten], bins=4000)
        # hist_soft, _ = np.histogram(shifted_flat[mask_soft_flatten], bins=4000)
        # hist_bone, _ = np.histogram(shifted_flat[mask_bone_flatten], bins=4000)
        # hist_air = hist_air / np.sum(mask_air)
        # hist_soft = hist_soft / np.sum(mask_soft)
        # hist_bone = hist_bone / np.sum(mask_bone)
        # prior_x_class_air += hist_air
        # prior_x_class_soft += hist_soft
        # prior_x_class_bone += hist_bone

        # Pad the data
        pad_size = 200 - ct_data.shape[2]
        if pad_size < 0:
            print("The z dimension of the data is larger than 200.")
            continue
        print("--> The shape is padded from", ct_data.shape, "to", (ct_data.shape[0], ct_data.shape[1], 200))
        mask_air = np.pad(mask_air, ((0, 0), (0, 0), (pad_size//2, pad_size//2)), 'constant', constant_values=1)
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
    prior_class_sum = prior_class_air + prior_class_soft + prior_class_bone
    prior_class_air = prior_class_air / prior_class_sum
    prior_class_soft = prior_class_soft / prior_class_sum
    prior_class_bone = prior_class_bone / prior_class_sum

    prior_x /= n_file # take each case as one distribution
    prior_x = prior_x / np.sum(prior_x)
    prior_x_class /= n_file

    # check whether the prior is sum to 1
    print("P_class:", np.sum(prior_class_air), np.sum(prior_class_soft), np.sum(prior_class_bone))
    print("P_class shape:", prior_class_air.shape, prior_class_soft.shape, prior_class_bone.shape)
    print("P_x:", np.sum(prior_x))
    print("P_x shape:", prior_x.shape)
    print("P_x_class:", np.sum(prior_x_class))
    print("P_x_class shape:", prior_x_class.shape)

    # save the prior

    save_folder = config["prior_folder"]
    # save_name = "prior_CT.npy"
    save_name = "prior_CT_n500p500.npy"
    prior_CT = {
        "prior_class":{
            "air": prior_class_air,
            "soft": prior_class_soft,
            "bone": prior_class_bone
        },
        "prior_x": prior_x,
        "prior_x_class": prior_x_class,
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
