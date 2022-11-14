import os
import time
import numpy as np
# from model import UNet_Theseus as UNet
# from model import UNETR_mT
# from monai.networks.nets.unet import UNet
from model import UNet_channelDO as UNet
from monai.networks.layers.factories import Act, Norm
from utils import iter_all_order, iter_some_order
from scipy.stats import mode

model_list = [
    
    # "Seg532_Unet_channnel_r050",
    # "Seg532_Unet_channnel_r050w",
    # "Seg532_Unet_channnel_r100",
    # "Seg532_Unet_channnel_r100w",
    # "Seg532_Unet"
    # "Seg532_UnetR_MC_D50_R100"

    ["Seg532_Unet_channnel_r050/", [0], False,], # gpu1
    ["Seg532_Unet_channnel_r050w/", [6], True,], # gpu6
    ["Seg532_Unet_channnel_r100/", [7], False,], # gpu7
    ["Seg532_Unet_channnel_r100w/", [7], True,], # gpu6
    
]


print("Model index: ", end="")
current_model_idx = int(input()) - 1
cmi = current_model_idx
print(model_list[current_model_idx])
time.sleep(1)

root_dir = model_list[cmi][0]
gpu_list = model_list[cmi][1]
is_WDO = model_list[cmi][2]

import os
# from monai.networks.nets.unet import UNet
# from model import UNet_Theseus as UNet
from monai.networks.layers.factories import Act, Norm
from utils import iter_all_order
from scipy.stats import mode

n_cls = 14
train_dict = {}
train_dict["root_dir"] = "./project_dir/"+root_dir+"/"
if not os.path.exists(train_dict["root_dir"]):
    os.mkdir(train_dict["root_dir"])
train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_532.json"
train_dict["gpu_list"] = gpu_list
train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2] # [2,2,2,2,2,2,2] for unet
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
val_files = load_decathlon_datalist(datasets, True, "test")

gpu_list = ','.join(str(x) for x in train_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "test")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

model = UNet( 
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=6,
    act=Act.PRELU,
    norm=Norm.INSTANCE,
    dropout=0.5,
    bias=True,
    is_WDO=is_WDO,
    ).to(device)

pre_train_state = {}
pre_train_model = torch.load(train_dict["root_dir"]+"best_metric_model.pth")
# print(pre_train_model.keys())

for model_key in model.state_dict().keys():
    pre_train_state[model_key] = pre_train_model[model_key]
     
model.load_state_dict(pre_train_state)

model.train()
order_list = iter_all_order(train_dict["alt_blk_depth"])
# order_list = iter_all_order([2,2,2,2,2,2,2,2,2])
order_list_cnt = len(order_list)


for case_num in range(6):
    # case_num = 4
    # model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image_meta_dict"]["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.from_numpy(np.expand_dims(img, 1)).float().cuda()
        val_labels = torch.from_numpy(np.expand_dims(label, 1)).float().cuda()

        _, _, ax, ay, az = val_labels.size()
        total_pixel = ax * ay * az
        output_array = np.zeros((ax, ay, az, order_list_cnt))
        for idx_bdo in range(order_list_cnt):
            # print(idx_bdo)
            # print(device)
            val_outputs = sliding_window_inference(
                val_inputs, [96, 96, 96], 8, model, overlap=1/8, device=device,
                mode="gaussian", sigma_scale=0.125, padding_mode="constant", # , order=order_list[idx_bdo],
            )
            output_array[:, :, :, idx_bdo] = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :]

        val_mode = np.asarray(np.squeeze(mode(output_array, axis=3).mode), dtype=int)

        for idx_diff in range(order_list_cnt):
            output_array[:, :, :, idx_diff] -= val_mode
        output_array = np.abs(output_array)
        output_array[output_array>0] = 1

        val_pct = np.sum(output_array, axis=3)/order_list_cnt


        # vote recorder
        # 128 list * 128 error
        for idx_vote in range(order_list_cnt):
            curr_path = order_list[idx_vote]
            curr_error = np.sum(output_array[:, :, :, idx_vote])/total_pixel
            for idx_path in range(len(train_dict["alt_blk_depth"])):
                # e.g. [*,*,1,*,*] then errors go to this list
                path_vote[idx_path][order_list[idx_vote][idx_path]].append(curr_error)

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_vote.npy"), 
            path_vote,
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_vote.npy"))



        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_vote.npy"), 
            val_inputs.cpu().numpy()[0, 0, :, :, :],
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_vote.npy"), 
            val_labels.cpu().numpy()[0, 0, :, :, :],
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_vote.npy"), 
            val_mode,
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_vote.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"), 
            val_pct,
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"))
