import os
import gc
import glob
import copy
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

import torch
import torchvision
import requests

# from model import SwinTransformer3D
from monai.inferers import sliding_window_inference
# from utils import sliding_window_inference
from utils import add_noise

# ==================== dict and config ====================

name_array = [
    "Iman_PETCT_v2",
]

for name in name_array:

    test_dict = {}
    test_dict = {}
    test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    test_dict["project_name"] = name # "Bayesian_MTGD_v2_unet_do10_MTGD15"
    test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
    test_dict["gpu_ids"] = [5]
    test_dict["eval_file_cnt"] = 1
    # test_dict["best_model_name"] = "model_best_193.pth"
    test_dict["eval_sample"] = 1
    test_dict["eval_save_folder"] = "pred_monai"

    train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]
    print("input size:", train_dict["input_size"])

    test_dict["seed"] = train_dict["seed"]
    test_dict["input_size"] = train_dict["input_size"]


    for path in [test_dict["save_folder"], test_dict["save_folder"]+test_dict["eval_save_folder"]]:
        if not os.path.exists(path):
            os.mkdir(path)

    np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)


    # ==================== basic settings ====================

    np.random.seed(test_dict["seed"])
    gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
    if "curr" in model_list[-1]:
        print("Remove model_best_curr")
        model_list.pop()
    target_model = model_list[-1]
    # target_model = test_dict["save_folder"]+test_dict["best_model_name"]
    model = torch.load(target_model, map_location=torch.device('cpu'))
    print("--->", target_model, " is loaded.")

    model = model.to(device)
    # loss_func = getattr(nn, train_dict['loss_term'])
    loss_func = nn.SmoothL1Loss()

    # ==================== data division ====================

    data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
    X_list = data_div['test_list_X']
    if test_dict["eval_file_cnt"] > 0:
        X_list = X_list[:test_dict["eval_file_cnt"]]

    # X_list = ["./data_dir/Iman_MR/norm/00550.nii.gz"]

    # ==================== Evaluating ====================

    # wandb.watch(model)

    file_list = X_list
    iter_tag = "test"
    cnt_total_file = len(file_list)
    model.eval()

    cnt_each_cube = 1
    cov_array = []

    for cnt_file, file_path in enumerate(file_list):
        
        x_path = file_path
        y_path = file_path.replace("X", "Y")
        file_name = os.path.basename(file_path)
        print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="") # 
        x_file = nib.load(x_path)
        y_file = nib.load(y_path)
        x_data = x_file.get_fdata()
        y_data = y_file.get_fdata()

        ax, ay, az = x_data.shape
        case_loss = 0

        input_data = np.expand_dims(x_data, (0,1))

        with torch.no_grad():
            y_hat = sliding_window_inference(
                    inputs = torch.from_numpy(input_data).float().to(device), 
                    roi_size = test_dict["input_size"], 
                    sw_batch_size = 4, 
                    predictor = model,
                    overlap=0.25, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    )

        output_data = y_hat.cpu().detach().numpy()
        print(output_data.shape)

        test_file = nib.Nifti1Image(np.squeeze(output_data), x_file.affine, x_file.header)
        test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii", ".nii.gz")
        nib.save(test_file, test_save_name)
        print(test_save_name)

        # test_file = nib.Nifti1Image(np.squeeze(x_data), x_file.affine, x_file.header)
        # test_save_name = train_dict["save_folder"]+test_dict["eval_save_folder"]+"/"+file_name.replace(".nii.gz", "_x.nii.gz")
        # nib.save(test_file, test_save_name)
        # print(test_save_name)


    # np.save("./metric_bayesian/"+test_dict["project_name"]+"_cov.npy", cov_array)
    # total_loss /= cnt_total_file
    # print("Total ", train_dict['loss_term'], total_loss)
    # np.save(train_dict["save_folder"]+"pred_monai/", os.path.basename(model_list[-1])+"_total_loss.npy", total_loss)



