import os
import time
import numpy as np
from model import UNet_Theseus as UNet
from model import UNETR_mT
from monai.networks.layers.factories import Act, Norm
from utils import iter_all_order, iter_some_order
from scipy.stats import mode

model_list = [
    ["./project_dir/JN_Unet_bdo_ab4/",      # project root folder
        [4,4,4,4,4,4,4],                    # alter block number of each block
        6,                                  # GPU number for using
        "",                                 # special tag for input/output files
        "Unet",                             # model type
        "dataset_0.json",                   # data division file
        256,],                              # how many predictios for one single case
    
    ["./project_dir/JN_Unet_bdo_ab2468642/",
        [2,4,6,8,6,4,2], 
        6, 
        "", 
        "Unet",
        "dataset_0.json",
        256,],

    ["./project_dir/Seg532_Unet_ab2/",
        [2,2,2,2,2,2,2], 
        5, 
        "", 
        "Unet",
        "dataset_532.json",
        0,],

    ["./project_dir/Seg532_UnetR_ab2/",
        [2,2,2,2,2,2,2,2,2,2], 
        5, 
        "", 
        "UnetR",
        "dataset_532.json",
        256,],
]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
cmi = current_model_idx
print(model_list[current_model_idx])
time.sleep(1)



n_cls = 14
config_dict = {}
config_dict["root_dir"] = model_list[cmi][0]
config_dict["alt_blk_depth"] = model_list[cmi][1]
config_dict["gpu_list"] = [model_list[cmi][2]]
config_dict["tag"] = model_list[cmi][3]
config_dict["model_type"] = model_list[cmi][4]
config_dict["split_JSON"] = model_list[cmi][5]
config_dict["num_trial"] = model_list[cmi][6]
np.save(config_dict["root_dir"]+"config_dict.npy", config_dict)

root_dir = config_dict["root_dir"]
print(root_dir)

if not os.path.exists(config_dict["root_dir"]):
    os.mkdir(config_dict["root_dir"])
config_dict["data_dir"] = "./data_dir/JN_BTCV/"
# config_dict["alt_blk_depth"] = [2,2,2,2,2,2,2] # for unet
# config_dict["alt_blk_depth"] = [2,2,2,2,2,2,2,2,2,2] # for unetR
root_dir = config_dict["root_dir"]
print(root_dir)

if config_dict["num_trial"] > 0:
    order_list, time_frame = iter_some_order(config_dict["alt_blk_depth"], config_dict["num_trial"])
else:
    order_list, time_frame = iter_all_order(config_dict["alt_blk_depth"])
order_list_cnt = len(order_list)
np.save(root_dir+"order_list_"+time_frame+".npy", order_list)

path_vote =[]
for idx in range(len(config_dict["alt_blk_depth"])):
    path_vote_sub = []
    for idx_sub in range(config_dict["alt_blk_depth"][idx]):
        path_vote_sub.append([])
    path_vote.append(path_vote_sub)


import os
import shutil
import tempfile

import matplotlib.pyplot as plt
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

print_config()

gpu_list = ','.join(str(x) for x in config_dict["gpu_list"])
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

data_dir = config_dict["data_dir"]
split_JSON = config_dict["split_JSON"]

datasets = data_dir + split_JSON
# datalist = load_decathlon_datalist(datasets, True, "training")
if config_dict["split_JSON"] == "dataset_0.json":
    val_files = load_decathlon_datalist(datasets, True, "validation")
if config_dict["split_JSON"] == "dataset_532.json":
    val_files = load_decathlon_datalist(datasets, True, "test")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

if config_dict["model_type"] == "Unet":

    model = UNet( 
        spatial_dims=3,
        in_channels=1,
        out_channels=n_cls,
        channels=(64, 128, 256, 512),
        strides=(2, 2, 2),
        num_res_units=6,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0.,
        bias=True,
        alter_block=config_dict["alt_blk_depth"],
        ).to(device)

if config_dict["model_type"] == "UnetR":

    model = UNETR_mT(
        in_channels=1,
        out_channels=n_cls,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
        alter_block=config_dict["alt_blk_depth"],
        ).to(device)


model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

slice_map = {
    "img0035.nii.gz": 170,
    "img0036.nii.gz": 230,
    "img0037.nii.gz": 204,
    "img0038.nii.gz": 204,
    "img0039.nii.gz": 204,
    "img0040.nii.gz": 180,
}

for case_num in range(6):
    # case_num = 4
    model.eval()
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
            print(idx_bdo)
            val_outputs = sliding_window_inference(
                val_inputs, (96, 96, 96), 8, model, overlap=0.25, order=order_list[idx_bdo],
            )
            output_array[:, :, :, idx_bdo] = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :]

        val_mode = np.asarray(np.squeeze(mode(output_array, axis=3).mode), dtype=int)

        # np.save(
        #     config_dict["root_dir"]+img_name.replace(".nii.gz", "_output_array.npy"), 
        #     output_array,
        # )
        # print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_output_array.npy"))
        # exit()

        # all to one hot for predictions
        # output_onehot = np.zeros((ax, ay, az, n_cls))
        # for idx_onehot in range(order_list_cnt):
        #     output_onehot += np.identity(n_cls)[np.asarray(output_array[:, :, :, idx_onehot], dtype=int)]
        # val_onehot = np.identity(n_cls)[np.asarray(val_mode, dtype=int)]
        # val_onehot_com = 1-val_onehot
        # print(output_onehot.shape, val_onehot.shape)
        # val_diff = np.abs(val_onehot*order_list_cnt-output_onehot)/order_list_cnt # how many votes in difference
        # val_diff = np.multiply(val_diff, val_onehot_com) # multiply with mask to remove correct votes
        # val_L1 = np.sum(val_diff, axis=3)
        # val_L2 = np.square(val_diff, axis=3)
        # val_L2 = np.sum(val_L2, axis=3)
        # val_L2 = np.aqrt(val_L2, axis=3)

        for idx_diff in range(order_list_cnt):
            output_array[:, :, :, idx_diff] -= val_mode
        output_array = np.abs(output_array)
        output_array[output_array>0] = 1

        val_pct = np.sum(output_array, axis=3)/order_list_cnt

        # vote recorder
        # 128 list * 128 error
        for idx_vote in range(order_list_cnt):
            curr_path = order_list[idxidx_vote]
            curr_error = np.sum(output_array[:, :, :, idx_vote])/total_pixel
            for idx_path in range(len(config_dict["alt_blk_depth"])):
                # e.g. [*,*,1,*,*] then errors go to this list
                path_vote[idx_path][order_list[idx_vote][idx_path]].append(curr_error)

        # path_vote
        np.save(
            config_dict["root_dir"]+img_name.replace(".nii.gz", ""+config_dict["tag"]+".npy"), 
            path_vote,
        )
        print(config_dict["root_dir"]+img_name.replace(".nii.gz", ""+config_dict["tag"]+".npy"))

        # val_inputs (X)
        np.save(
            config_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"), 
            val_inputs.cpu().numpy()[0, 0, :, :, :],
        )
        print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"))

        # val_labels (Y)
        np.save(
            config_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"), 
            val_labels.cpu().numpy()[0, 0, :, :, :],
        )
        print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"))

        # val_mode (Z) from mode of all predictions
        np.save(
            config_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"), 
            val_mode,
        )
        print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"))

        # val_pct (Z_pct) percentage of mode in all predictions
        np.save(
            config_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"), 
            val_pct,
        )
        print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0"+config_dict["tag"]+".npy"))


        # np.save(
        #     config_dict["root_dir"]+img_name.replace(".nii.gz", "_L1_RAS_1.5_1.5_2.0.npy"), 
        #     val_L1,
        # )
        # print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_L1_RAS_1.5_1.5_2.0.npy"))

        # np.save(
        #     config_dict["root_dir"]+img_name.replace(".nii.gz", "_L2_RAS_1.5_1.5_2.0.npy"), 
        #     val_L2,
        # )
        # print(config_dict["root_dir"]+img_name.replace(".nii.gz", "_L2_RAS_1.5_1.5_2.0.npy"))

        # quick view images
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.imshow(
            torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
        )
        # plt.show()
        plt.savefig(config_dict["root_dir"]+"JN"+config_dict["tag"]+"_{}.png".format(img_name), dpi=300)