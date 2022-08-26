import os
# from monai.networks.nets.unet import UNet
from model import UNet_Theseus as UNet
from monai.networks.layers.factories import Act, Norm
from utils import iter_all_order

n_cls = 14
train_dict = {}
train_dict["root_dir"] = "./project_dir/JN_Unet_bdo/"
if not os.path.exists(train_dict["root_dir"]):
    os.mkdir(train_dict["root_dir"])
train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_0.json"
train_dict["gpu_list"] = [6]
train_dict["alt_blk_depth"] = [2] # [2,2,2,2,2,2,2] for unet
# train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2,2,2] # [2,2,2,2,2,2,2,2,2] for unet

import os
import gc
import copy
import glob
import time
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

from monai.inferers import sliding_window_inference

import torch
from monai.data import (
    load_decathlon_datalist,
    decollate_batch,
)

root_dir = train_dict["root_dir"]
print(root_dir)

data_dir = train_dict["data_dir"]
split_JSON = train_dict["split_JSON"]

datasets = data_dir + split_JSON
val_files = load_decathlon_datalist(datasets, True, "validation")

gpu_list = ','.join(str(x) for x in train_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet( 
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=6,
    act=Act.PRELU,
    norm=Norm.INSTANCE,
    dropout=0.,
    bias=True,
    ).to(device)

pre_train_state = {}
pre_train_model = torch.load(train_dict["root_dir"]+"best_metric_model.pth")

for model_key in model.state_dict().keys():
    pre_train_state[model_key] = pre_train_model[model_key]
     
model.load_state_dict(pre_train_state)

model.eval()
order_list = iter_all_order(train_dict["alt_blk_depth"])
# order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
order_list_cnt = len(order_list)
with torch.no_grad():
    for idx, val_tuple in enumerate(val_files):
        img_path = val_tuple['image']
        lab_path = val_tuple['label']
        file_name = os.path.basename(lab_path)
        input_data = nib.load(img_path).get_fdata()
        lab_file = nib.load(lab_path)
        ax, ay, az = input_data.shape
        output_array = np.zeros((order_list_cnt, ax, ay, az))

        input_data = np.expand_dims(input_data, (0,1))
        input_data = torch.from_numpy(input_data).float().to(device)
        for idx_bdo in range(order_list_cnt):
            # print(idx_bdo)
            y_hat = sliding_window_inference(
                    inputs = input_data, 
                    roi_size = [96, 96, 96], 
                    sw_batch_size = 4, 
                    predictor = model,
                    overlap=0.25, 
                    mode="gaussian", 
                    sigma_scale=0.125, 
                    padding_mode="constant", 
                    cval=0.0, 
                    sw_device=device, 
                    device=device,
                    # order=order_list[idx_bdo],
                    )
            print(y_hat.shape)
            np.save("raw_output.npy", y_hat)
            y_hat = nn.Softmax(dim=1)(y_hat).cpu().detach().numpy()
            y_hat = np.argmax(np.squeeze(y_hat), axis=0)
            # print(y_hat.shape)
            output_array[idx_bdo, :, :, :] = y_hat

        val_median = np.median(output_array, axis=0)
        val_std = np.std(output_array, axis=0)

        # test_file = nib.Nifti1Image(np.squeeze(val_median), lab_file.affine, lab_file.header)
        # test_save_name = train_dict["root_dir"]+file_name.replace(".nii.gz", "_pred.nii.gz")
        # nib.save(test_file, test_save_name)
        # print(test_save_name)

        # test_file = nib.Nifti1Image(np.squeeze(val_std), lab_file.affine, lab_file.header)
        # test_save_name = train_dict["root_dir"]+file_name.replace(".nii.gz", "_std.nii.gz")
        # nib.save(test_file, test_save_name)
        # print(test_save_name)