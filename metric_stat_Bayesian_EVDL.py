import numpy as np
import glob
import os

results_folder = "results/Bayesian_EVDL"
results_list = sorted(glob.glob(os.path.join(results_folder, "*.nii.gz.npy")))

error_std_corr = []
error_bay_corr = []
error_evdl_corr = []

for result_file in results_list:
    print("Loading", result_file)
    result = np.load(result_file, allow_pickle=True).item()
    #     data = {
    #     "error_std_corr": error_std_corr,
    #     "error_bay_corr": error_bay_corr,
    #     "error_evdl_corr": error_evdl_corr
    # }
    print("error_std_corr", result["error_std_corr"].shape)
    print("error_bay_corr", result["error_bay_corr"].shape)
    print("error_evdl_corr", result["error_evdl_corr"].shape)
    print()
    error_std_corr.append(result["error_std_corr"])
    error_bay_corr.append(result["error_bay_corr"])
    error_evdl_corr.append(result["error_evdl_corr"])

error_std_corr = np.mean(np.array(error_std_corr))
error_bay_corr = np.mean(np.array(error_bay_corr))
error_evdl_corr = np.mean(np.array(error_evdl_corr))
print("Loaded all results")
print("error_std_corr", error_std_corr.shape)
print("error_bay_corr", error_bay_corr.shape)
print("error_evdl_corr", error_evdl_corr.shape)

corr_dict = {
    "error_std_corr": error_std_corr,
    "error_bay_corr": error_bay_corr,
    "error_evdl_corr": error_evdl_corr
}

np.save(os.path.join(results_folder, "corr_dict.npy"), corr_dict)
print("Correlation saved to", os.path.join(results_folder, "corr_dict.npy"))