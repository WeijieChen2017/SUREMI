# Imports
import os
import glob
import time
import torch
import numpy as np
import nibabel as nib

from monai.inferers import sliding_window_inference
from model import UNet_MDO as UNet
from utils import iter_all_order

import os
import numpy as np
import nibabel as nib
from monai.inferers import sliding_window_inference
from utils import iter_all_order

from matplotlib import pyplot as plt
from scipy.stats import norm

def add_gaussian_noise(mr_volume, std_level):
    noise = np.random.normal(0, std_level, mr_volume.shape)
    noisy_mr_volume = mr_volume + noise
    noisy_mr_volume = np.clip(noisy_mr_volume, 0, 1)
    noisy_mr_volume = noisy_mr_volume.astype(np.float32)
    return noisy_mr_volume

def add_rician_noise(mr_volume, std_level):
    noise = np.random.normal(0, std_level, mr_volume.shape)
    noisy_mr_volume = np.sqrt((mr_volume + noise) ** 2 + noise ** 2)
    noisy_mr_volume = np.clip(noisy_mr_volume, 0, 1)
    noisy_mr_volume = noisy_mr_volume.astype(np.float32)
    return noisy_mr_volume

def add_rayleigh_noise(mr_volume, std_level):
    noise = np.random.rayleigh(std_level, mr_volume.shape)
    noisy_mr_volume = mr_volume + noise
    noisy_mr_volume = np.clip(noisy_mr_volume, 0, 1)
    noisy_mr_volume = noisy_mr_volume.astype(np.float32)
    return noisy_mr_volume

def add_salt_and_pepper_noise(mr_volume, noise_level):
    noisy_mr_volume = mr_volume.copy()
    mask = np.random.rand(*mr_volume.shape) < noise_level
    noisy_mr_volume[mask] = 0
    mask = np.random.rand(*mr_volume.shape) < noise_level
    noisy_mr_volume[mask] = 1
    noisy_mr_volume = noisy_mr_volume.astype(np.float32)
    return noisy_mr_volume

def radial_trajectory(n_spokes, n_points):
    # Create an empty k-space plane
    k_space = np.zeros((n_points, n_points), dtype=np.float16)
    
    # Calculate the angles for each spoke
    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False)
    
    for angle in angles:
        # Calculate the coordinates for this spoke
        x = np.linspace(-n_points//2, n_points//2-1, n_points) * np.cos(angle)
        y = np.linspace(-n_points//2, n_points//2-1, n_points) * np.sin(angle)
        
        # Ensure indices are within the valid range
        x_indices = np.clip(np.round(x + n_points // 2).astype(int), 0, n_points - 1)
        y_indices = np.clip(np.round(y + n_points // 2).astype(int), 0, n_points - 1)
        
        # Simulate the acquisition by setting these points to 1 (or some other arbitrary value)
        k_space[x_indices, y_indices] = 1

    return k_space

def radial_sample_mr_image(mr_volume, n_spokes, n_points):
    mr_volume_fft = np.fft.fftn(mr_volume)
    # fftshift
    mr_volume_fft = np.fft.fftshift(mr_volume_fft)
    k_space_radial = radial_trajectory(n_spokes, n_points)
    sample_ratio = np.sum(k_space_radial) / k_space_radial.size
    undersampled_mr_volume_fft = mr_volume_fft * k_space_radial
    # ifftshift
    undersampled_mr_volume_fft = np.fft.ifftshift(undersampled_mr_volume_fft)
    undersampled_mr_volume = np.fft.ifftn(undersampled_mr_volume_fft)
    undersampled_mr_volume = np.abs(undersampled_mr_volume)
    undersampled_mr_volume = undersampled_mr_volume.astype(np.float32)
    return undersampled_mr_volume, sample_ratio

def dense_spiral_trajectory(n_turns, n_points_per_turn):
    # Total number of points is n_turns * n_points_per_turn
    total_points = n_turns * n_points_per_turn
    n_pixel = 256
    # Create an empty k-space plane
    k_space = np.zeros((n_pixel, n_pixel), dtype=np.float16)  # Increased size for better visualization

    # Define the spiral trajectory
    theta = np.linspace(0, 2 * np.pi * n_turns, total_points)
    r = np.linspace(0, 256, total_points)  # Adjust radius to fill the k-space more appropriately
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Convert polar coordinates to Cartesian indices
    x_indices = np.clip(np.round(x + n_pixel // 2).astype(int), 0, n_pixel-1)  # Adjust centering based on new k-space size
    y_indices = np.clip(np.round(y + n_pixel // 2).astype(int), 0, n_pixel-1)
    
    # Simulate the acquisition by marking these points
    for i in range(len(x_indices)):
        k_space[x_indices[i], y_indices[i]] = 1

    return k_space

def spiral_sample_mr_image(mr_volume, n_turns, n_points_per_turn):
    mr_volume_fft = np.fft.fftn(mr_volume)
    # fftshift
    mr_volume_fft = np.fft.fftshift(mr_volume_fft)
    k_space_spiral = dense_spiral_trajectory(n_turns, n_points_per_turn)
    undersampled_mr_volume_fft = mr_volume_fft * k_space_spiral
    sample_ratio = np.sum(k_space_spiral) / k_space_spiral.size
    # ifftshift
    undersampled_mr_volume_fft = np.fft.ifftshift(undersampled_mr_volume_fft)
    undersampled_mr_volume = np.fft.ifftn(undersampled_mr_volume_fft)
    undersampled_mr_volume = np.abs(undersampled_mr_volume)
    undersampled_mr_volume = undersampled_mr_volume.astype(np.float32)
    return undersampled_mr_volume, sample_ratio

Gaussian_level = np.asarray([10, 20, 50, 100, 200])/3000
Rician_level = np.asarray([10, 20, 50, 100, 200])/3000
Rayleigh_level = np.asarray([10, 20, 50, 100, 200])/3000
Salt_and_pepper_level = np.asarray([0.01, 0.02, 0.05, 0.1, 0.2])
Radial_sampling_parameters = [(300, 256), (240, 256), (180, 256), (120, 256), (60, 256)]
Spiral_sampling_parameters = [(240, 300), (210, 300), (210, 240), (180, 240), (180, 180)]

# Configuration dictionary
default_config = {
    "model_list": [
        "Theseus_v2_181_200_rdp1",
    ],
    "project_name": "Theseus_v2_181_200_rdp1",
    "special_cases": [],
    "gpu_ids": [0],
    "eval_file_cnt": 0,
    "eval_save_folder": "corrpution",
    "save_tag": "_corp",
    "stride_division": 8,
    "alt_blk_depth": [2, 2, 2, 2, 2, 2, 2],
    # "alt_blk_depth": [2, 2, 2, 2, 2, 2, 2],
    "pad_size": 0,
}

# Function to calculate weighted mean and std for a segment of the histogram
def calculate_statistics(counts, midpoints):
    mean = np.sum(midpoints * counts) / np.sum(counts)
    variance = np.sum(counts * (midpoints - mean) ** 2) / np.sum(counts)
    std = np.sqrt(variance)
    return mean, std

def setup_environment(config):
    np.random.seed(config["seed"])
    gpu_list = ','.join(str(x) for x in config["gpu_ids"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    import torch
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

def evaluate_corruption(mr_volume, noise_level, corruption_type):
    if corruption_type == "Gaussian":
        noisy_mr_volume = add_gaussian_noise(mr_volume, noise_level)
    elif corruption_type == "Rician":
        noisy_mr_volume = add_rician_noise(mr_volume, noise_level)
    elif corruption_type == "Rayleigh":
        noisy_mr_volume = add_rayleigh_noise(mr_volume, noise_level)
    elif corruption_type == "Salt_and_pepper":
        noisy_mr_volume = add_salt_and_pepper_noise(mr_volume, noise_level)
    elif corruption_type == "Radial":
        noisy_mr_volume, sample_ratio = radial_sample_mr_image(mr_volume, *noise_level)
    elif corruption_type == "Spiral":
        noisy_mr_volume, sample_ratio = spiral_sample_mr_image(mr_volume, *noise_level)
    
    return noisy_mr_volume

def evalute_mr_output_median_std(mr_volume, model, device, config, file_name, idx, n_file, x_file, noise_level, corruption_type):

    ax, ay, az = input_data.shape
    input_data = np.expand_dims(mr_volume, (0, 1))
    input_data = torch.from_numpy(mr_volume).float().to(device)

    order_list, _ = iter_all_order(config["alt_blk_depth"])
    order_list_cnt = len(order_list)
    output_array = np.zeros((order_list_cnt, ax, ay, az))

    print("Processing: ", corruption_type, noise_level)
    print(input_data.shape)
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

    output_median = np.median(output_array, axis=0)
    output_std = np.std(output_array, axis=0)
    output_median = output_median
    
    print(f"[{idx+1}]/[{n_file}]: Processed: {file_name} for {corruption_type} noise level: {noise_level}")

    # Save processed data
    save_tag = f"{config['save_tag']}_{corruption_type}_{noise_level}"
    save_processed_data(output_median, x_file, file_name, config, f"{save_tag}_median")
    save_processed_data(output_std, x_file, file_name, config, f"{save_tag}_std")

def process_data(file_list, model, device, config):
    """
    Process a list of data files for inference with the given model and save the outputs.

    Parameters:
    - file_list: A list of paths to the input files.
    - model: The trained model for inference.
    - device: The device (CPU or GPU) to run the inference on.
    - config: Dictionary containing configuration and parameters for processing.
    """

    # output the corruption configuration into the txt file
    with open(os.path.join(config["save_folder"], config["eval_save_folder"], "corruption_config.txt"), "w") as f:
        f.write("Gaussian_level: ")
        f.write(str(Gaussian_level))
        f.write("\n")
        f.write("Rician_level: ")
        f.write(str(Rician_level))
        f.write("\n")
        f.write("Rayleigh_level: ")
        f.write(str(Rayleigh_level))
        f.write("\n")
        f.write("Salt_and_pepper_level: ")
        f.write(str(Salt_and_pepper_level))
        f.write("\n")
        f.write("Radial_sampling_parameters: ")
        f.write(str(Radial_sampling_parameters))
        f.write("\n")
        f.write("Spiral_sampling_parameters: ")
        f.write(str(Spiral_sampling_parameters))
        f.write("\n")

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

        # Add noise to the input data
        for noise_level in Gaussian_level:
            noisy_mr_volume = evaluate_corruption(input_data, noise_level, "Gaussian")
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, noise_level, "Gaussian")
        for noise_level in Rician_level:
            noisy_mr_volume = evaluate_corruption(input_data, noise_level, "Rician")
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, noise_level, "Rician")
        for noise_level in Rayleigh_level:
            noisy_mr_volume = evaluate_corruption(input_data, noise_level, "Rayleigh")
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, noise_level, "Rayleigh")
        for noise_level in Salt_and_pepper_level:
            noisy_mr_volume = evaluate_corruption(input_data, noise_level, "Salt_and_pepper")
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, noise_level, "Salt_and_pepper")
        for noise_level in Radial_sampling_parameters:
            noisy_mr_volume, sample_ratio = radial_sample_mr_image(input_data, *noise_level)
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, sample_ratio, "Radial")
        for noise_level in Spiral_sampling_parameters:
            noisy_mr_volume, sample_ratio = spiral_sample_mr_image(input_data, *noise_level)
            evalute_mr_output_median_std(noisy_mr_volume, model, device, config, file_name, idx, n_file, x_file, sample_ratio, "Spiral")

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
