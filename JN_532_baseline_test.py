import os
from monai.networks.nets.unet import UNet
# from model import UNet_Theseus as UNet
from monai.networks.layers.factories import Act, Norm
from utils import iter_all_order
from scipy.stats import mode


import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

import torch
import time

n_cls = 14
train_dict = {}
model_group = [
[
    [1001, [2],],
    [1002, [2],],
    [1003, [2],],
    [1004, [2],],
    [1005, [2],],
    [1006, [2],],
    [1007, [2],],
    [1008, [2],],
],
[
    [1009, [2],],
    [1010, [2],],
    [1011, [2],],
    [1012, [2],],
    [1013, [2],],
    [1013, [2],],
    [1014, [2],],
    [1015, [2],],
],
[
    [1016, [3],],
    [1017, [3],],
    [1018, [3],],
    [1019, [3],],
    [1020, [3],],
    [1021, [3],],
    [1022, [3],],
    [1023, [3],],
],
[
    [1024, [3],],
    [1025, [3],],
    [1026, [3],],
    [1027, [3],],
    [1028, [3],],
    [1029, [3],],
    [1030, [3],],
    [1031, [3],],
    [1032, [3],],
]
]

print("Model group index: ", end="")
current_model_group_idx = int(input()) - 1
print(model_group[current_model_group_idx])
time.sleep(1)

model_list = model_group[current_model_group_idx]

train_dict["gpu_list"] = model_list[0][1]
gpu_list = ','.join(str(x) for x in train_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.cuda.device(2)

train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_532.json"

data_dir = train_dict["data_dir"]
split_JSON = train_dict["split_JSON"]

datasets = data_dir + split_JSON
val_files = load_decathlon_datalist(datasets, True, "test")
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

for idx_sub_group in range(8):
    seed = model_list[idx_sub_group][0]
    folder_name = str(seed)
    # gpu = model_list[idx_sub_group][1]
    # Seg532_Unet_seed1016
    train_dict["root_dir"] = "./project_dir/Seg532_Unet_seed"+folder_name+"/"
    # if not os.path.exists(train_dict["root_dir"]):
    #     os.mkdir(train_dict["root_dir"])
    root_dir = train_dict["root_dir"]
    print(root_dir)

    # train_dict["gpu_list"] = gpu
    train_dict["alt_blk_depth"] = [1]
    # train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2] # [2,2,2,2,2,2,2] for unet
    # train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2,2,2] # [2,2,2,2,2,2,2,2,2] for unet

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
        )#.to(device).load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

    pre_train_state = {}
    pre_train_model = torch.load(train_dict["root_dir"]+"best_metric_model.pth")

    for model_key in model.state_dict().keys():
        pre_train_state[model_key] = pre_train_model[model_key]
         
    model.load_state_dict(pre_train_state)

    model.eval()
    model.to(device)
    # model.train()
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
            val_inputs = torch.from_numpy(np.expand_dims(img, 1)).float().to(device)
            val_labels = torch.from_numpy(np.expand_dims(label, 1)).float().to(device)

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

            # for idx_diff in range(order_list_cnt):
            #     output_array[:, :, :, idx_diff] -= val_mode
            # output_array = np.abs(output_array)
            # output_array[output_array>0] = 1

            # val_pct = np.sum(output_array, axis=3)/order_list_cnt


            # vote recorder
            # 128 list * 128 error
            # for idx_vote in range(order_list_cnt):
            #     curr_path = order_list[idx_vote]
            #     curr_error = np.sum(output_array[:, :, :, idx_vote])/total_pixel
            #     for idx_path in range(len(train_dict["alt_blk_depth"])):
            #         # e.g. [*,*,1,*,*] then errors go to this list
            #         path_vote[idx_path][order_list[idx_vote][idx_path]].append(curr_error)

            # np.save(
            #     train_dict["root_dir"]+img_name.replace(".nii.gz", "_vote.npy"), 
            #     path_vote,
            # )
            # print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_vote.npy"))



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

            # np.save(
            #     train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"), 
            #     val_pct,
            # )
            # print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_vote.npy"))








# with torch.no_grad():
#     for idx, val_tuple in enumerate(val_files):
#         img_path = val_tuple['image']
#         lab_path = val_tuple['label']
#         file_name = os.path.basename(lab_path)
#         input_data = nib.load(img_path).get_fdata()
#         lab_file = nib.load(lab_path)
#         ax, ay, az = input_data.shape
#         output_array = np.zeros((order_list_cnt, ax, ay, az))

#         # ScaleIntensityRanged(
#         #     keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
#         # ),
#         a_min=-175
#         a_max=250
#         b_min=0.0
#         b_max=1.0
#         input_data = (input_data - a_min) / (a_max - a_min)
#         input_data[input_data > 1.] = 1.
#         input_data[input_data < 0.] = 0.

#         input_data = np.expand_dims(input_data, (0,1))
#         input_data = torch.from_numpy(input_data).float().to(device)
#         for idx_bdo in range(order_list_cnt):
#             y_hat = sliding_window_inference(
#                     inputs = input_data, 
#                     roi_size = [96, 96, 96], 
#                     sw_batch_size = 4, 
#                     predictor = model,
#                     overlap=0.25, 
#                     mode="gaussian", 
#                     sigma_scale=0.125, 
#                     padding_mode="constant", 
#                     cval=0.0, 
#                     sw_device=device, 
#                     device=device,
#                     # order=order_list[idx_bdo],
#                     )
#             print(y_hat.shape)
#             # np.save("raw_output.npy", y_hat.cpu().detach().numpy())
#             # exit()
#             y_hat = nn.Softmax(dim=1)(y_hat).cpu().detach().numpy()
#             y_hat = np.argmax(np.squeeze(y_hat), axis=0)
#             print(np.unique(y_hat))
#             output_array[idx_bdo, :, :, :] = y_hat

#         # val_median = np.median(output_array, axis=0)
#         # val_std = np.std(output_array, axis=0)
#         val_mode = np.squeeze(mode(output_array, axis=0).mode)
#         print(np.unique(val_mode))
#         val_std = np.zeros((val_mode.shape))
#         for idx_std in range(order_list_cnt):
#             val_std += np.square(output_array[idx_std, :, :, :]-val_mode)
#             print(np.mean(val_std))
#         val_std = np.sqrt(val_std) 

#         test_file = nib.Nifti1Image(np.squeeze(val_mode), lab_file.affine, lab_file.header)
#         test_save_name = train_dict["root_dir"]+file_name.replace(".nii.gz", "_pred_seg.nii.gz")
#         nib.save(test_file, test_save_name)
#         print(test_save_name)

#         test_file = nib.Nifti1Image(np.squeeze(val_std), lab_file.affine, lab_file.header)
#         test_save_name = train_dict["root_dir"]+file_name.replace(".nii.gz", "_std_seg.nii.gz")
#         nib.save(test_file, test_save_name)
#         print(test_save_name)