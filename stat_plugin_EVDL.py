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
default_config = {
    "model_list": [
        "Theseus_v2_181_200_rdp1",
    ],
    "project_name": "Theseus_v2_181_200_rdp1",
    "special_cases": [],
    "gpu_ids": [0],
    "eval_file_cnt": 0,
    "eval_save_folder": "analysis",
    "save_tag": "_EVDL",
    "stride_division": 8,
    "alt_blk_depth": [2, 2, 2, 2, 2, 2, 2],
    # "alt_blk_depth": [2, 2, 2, 2, 2, 2, 2],
    "pad_size": 0,

}

def setup_environment(config):
    np.random.seed(config["seed"])
    gpu_list = ','.join(str(x) for x in config["gpu_ids"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_model(config, device):
    model_list = sorted(glob.glob(os.path.join(config["save_folder"], "model_best_*.pth")))
    if "curr" in model_list[-1]:
        model_list.pop()
    target_model = model_list[-1]
    model = torch.load(target_model, map_location=torch.device('cpu')).to(device)
    print("--->", target_model, " is loaded.")
    return model

def process_data(file_list, model, device, config):
    """
    Process a list of data files for inference with the given model and save the outputs.

    Parameters:
    - file_list: A list of paths to the input files.
    - model: The trained model for inference.
    - device: The device (CPU or GPU) to run the inference on.
    - config: Dictionary containing configuration and parameters for processing.
    """
    n_file = len(file_list)
    for idx, file_path in enumerate(file_list):
        print(f"[{idx+1}]/[{n_file}]: Processing: {file_path}")
        x_path = file_path
        file_name = os.path.basename(file_path)

        # Load data
        x_file = nib.load(x_path)
        x_data = x_file.get_fdata()

        # Prepare data
        ax, ay, az = x_data.shape
        if config["pad_size"] > 0:
            input_data = np.pad(x_data, ((config["pad_size"], config["pad_size"]), (config["pad_size"], config["pad_size"]), (config["pad_size"], config["pad_size"])), 'constant')
        else:
            input_data = x_data
        input_data = np.expand_dims(input_data, (0, 1))
        input_data = torch.from_numpy(input_data).float().to(device)

        order_list, _ = iter_all_order(config["alt_blk_depth"])
        order_list_cnt = len(order_list)
        output_array = np.zeros((order_list_cnt, ax, ay, az))

        for idx_es in range(order_list_cnt):
            with torch.no_grad():
                y_hat = sliding_window_inference(
                    inputs=input_data, 
                    roi_size=config["input_size"], 
                    sw_batch_size=64, 
                    predictor=model,
                    overlap=1/config["stride_division"], 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    order=order_list[idx_es],
                )
                if config["pad_size"] > 0:
                    output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()[:, :, config["pad_size"]:-config["pad_size"], config["pad_size"]:-config["pad_size"], config["pad_size"]:-config["pad_size"]]
                else:
                    output_array[idx_es, :, :, :] = y_hat.cpu().detach().numpy()[:, :, :, :, :]

        # Post-process and analyze results (this part will depend on your specific needs, like calculating statistics or specific transformations)

        # evidence learning
        print("------>Evidential learning:")
        output_median = np.median(output_array, axis=0)
        # Now given the output_data, we perform the evidential learning to determine the uncertainty of the model
        output_mean = np.mean(output_array, axis=0)
        output_std = np.std(output_array, axis=0)
        output_th = output_mean + output_std # this is the threshold for the model to ne high or low range
        output_isHigh = output_median > output_th
        output_isLow = 1 - output_isHigh
        # in each pixel, count how many events are high and low
        output_massHigh = output_array > output_th.reshape((1, output_th.shape[0], output_th.shape[1], output_th.shape[2]))
        output_massHigh = np.sum(output_massHigh, axis=0) / order_list_cnt
        output_massLow = 1 - output_massHigh
        # for each pixel, if it is high, unc = std * sqrt(massHigh), if it is low, unc = std * sqrt(massLow)
        output_unc = output_std * np.sqrt(output_massHigh) * output_isHigh + output_std * np.sqrt(output_massLow) * output_isLow
        # save the uncertainty
        save_processed_data(output_median, x_file, file_name, config, tag="_median")
        save_processed_data(output_unc, x_file, file_name, config, tag="_unc_EVDL")
        # save_processed_data(output_isHigh, x_file, file_name, config, tag="_isHigh_EVDL")
        # save_processed_data(output_isLow, x_file, file_name, config, tag="_isLow_EVDL")
        # save_processed_data(output_massHigh, x_file, file_name, config, tag="_massHigh_EVDL")
        # save_processed_data(output_massLow, x_file, file_name, config, tag="_massLow_EVDL")
        save_processed_data(output_std, x_file, file_name, config, tag="_std_EVDL")
        save_processed_data(output_mean, x_file, file_name, config, tag="_mean_EVDL")

        # Bayesian learning
        # print("------>Bayesian learning:")
        # # -1000, air, -500, soft tissue, 250, bone, 3000, normalized by 4000, shifted by 1000
        # bone_mask = output_median > (250+1000)/4000
        # air_mask = output_median < (-500+1000)/4000
        # soft_tissue_mask = 1 - bone_mask - air_mask
        # # compute the mean and std for each tissue
        # bone_mean = np.mean(output_median[bone_mask])
        # bone_std = np.std(output_median[bone_mask])
        # air_mean = np.mean(output_median[air_mask])
        # air_std = np.std(output_median[air_mask])
        # soft_tissue_mean = np.mean(output_median[soft_tissue_mask])
        # soft_tissue_std = np.std(output_median[soft_tissue_mask])

        print(f"[{idx+1}]/[{n_file}]: Processed: {file_name}")

def save_processed_data(data, x_file, file_name, config, tag):
    """
    Save the processed data to a NIFTI file.

    Parameters:
    - data: The data array to save.
    - x_file: The original Nifti1Image file, used for affine and header information.
    - file_name: The base name of the file to save.
    - config: Configuration dictionary with save paths and settings.
    - tag: A tag to append to the file name for identification.
    """
    test_save_folder = os.path.join(config["save_folder"], config["eval_save_folder"])
    if not os.path.exists(test_save_folder):
        os.makedirs(test_save_folder)

    test_file = nib.Nifti1Image(np.squeeze(data), x_file.affine, x_file.header)
    test_save_name = os.path.join(test_save_folder, file_name.replace(".nii.gz", f"{tag}.nii.gz"))
    nib.save(test_file, test_save_name)
    print(f"Saved: {test_save_name}")

def main():
    train_dict = np.load("./project_dir/"+default_config["project_name"]+"/"+"dict.npy", allow_pickle=True)[()]
    config = {}
    config.update(train_dict)  # Update config with training dictionary parameters
    config["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    config["save_folder"] = "./project_dir/"+default_config["project_name"]+"/"
    config.update(default_config)

    device = setup_environment(config)
    print("Device:", device)
    model = load_model(config, device)
    
    # Data division
    data_div = np.load(os.path.join(config["save_folder"], "data_division.npy"), allow_pickle=True).item()
    X_list = data_div['test_list_X']
    if config["eval_file_cnt"] > 0:
        X_list = X_list[:config["eval_file_cnt"]]
    X_list.sort()

    # Populate the file_list based on special cases or use all files
    file_list = []
    if len(config["special_cases"]) > 0:
        for case_name in X_list:
            for spc_case_name in config["special_cases"]:
                if spc_case_name in os.path.basename(case_name):
                    file_list.append(case_name)
    else:
        file_list = X_list
    process_data(file_list, model, device, config)

if __name__ == "__main__":
    main()
