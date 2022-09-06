import os
# from model import UNet_Theseus as UNet
# from monai.networks.layers.factories import Act, Norm
from model import UNETR_bdo as UNETR
from model import UNETR_mT
from utils import iter_all_order, iter_some_order
from scipy.stats import mode
import numpy as np

n_cls = 14
train_dict = {}
train_dict["root_dir"] = "./project_dir/JN_UnetR_bdo/"
if not os.path.exists(train_dict["root_dir"]):
    os.mkdir(train_dict["root_dir"])
train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_0.json"
train_dict["gpu_list"] = [6]
# train_dict["alt_blk_depth"] = [4,2,2,2,2,1,1,1,1,1]
# train_dict["alt_blk_depth"] = [4,1,1,1,1,2,2,2,2,2]
train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2,2,2,2]
# train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2] # [2,2,2,2,2,2,2] for unet
# train_dict["alt_blk_depth"] = [2,2,2,2,2,2,2,2,2] # [2,2,2,2,2,2,2,2,2] for unet
# JN_UnetR_mT_4222211111
# JN_UnetR_mT_4111122222
root_dir = train_dict["root_dir"]
print(root_dir)

order_list, time_frame = iter_some_order(train_dict["alt_blk_depth"], 63)
order_list_cnt = len(order_list)
np.save(root_dir+"order_list_"+time_frame+".npy", order_list)

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
# from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()

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

data_dir = "./data_dir/JN_BTCV/"
split_JSON = "dataset_0.json"

datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


# model = UNETR(
#     in_channels=1,
#     out_channels=14,
#     img_size=(96, 96, 96),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)
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
    alter_block=train_dict["alt_blk_depth"],
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
        output_array = np.zeros((ax, ay, az, order_list_cnt))
        for idx_bdo in range(order_list_cnt):
            print(idx_bdo)
            val_outputs = sliding_window_inference(
                val_inputs, (96, 96, 96), 8, model, overlap=0.25, order=order_list[idx_bdo],
            )
            output_array[:, :, :, idx_bdo] = torch.argmax(val_outputs, dim=1).detach().cpu().numpy()[0, :, :, :]

        val_mode = np.asarray(np.squeeze(mode(output_array, axis=3).mode), dtype=int)

        # np.save(
        #     train_dict["root_dir"]+img_name.replace(".nii.gz", "_output_array.npy"), 
        #     output_array,
        # )
        # print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_output_array.npy"))
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


        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_64.npy"), 
            val_inputs.cpu().numpy()[0, 0, :, :, :],
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_x_RAS_1.5_1.5_2.0_64.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_64.npy"), 
            val_labels.cpu().numpy()[0, 0, :, :, :],
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_y_RAS_1.5_1.5_2.0_64.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_64.npy"), 
            val_mode,
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_z_RAS_1.5_1.5_2.0_64.npy"))

        np.save(
            train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_64.npy"), 
            val_pct,
        )
        print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_pct_RAS_1.5_1.5_2.0_64.npy"))


        # np.save(
        #     train_dict["root_dir"]+img_name.replace(".nii.gz", "_L1_RAS_1.5_1.5_2.0.npy"), 
        #     val_L1,
        # )
        # print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_L1_RAS_1.5_1.5_2.0.npy"))

        # np.save(
        #     train_dict["root_dir"]+img_name.replace(".nii.gz", "_L2_RAS_1.5_1.5_2.0.npy"), 
        #     val_L2,
        # )
        # print(train_dict["root_dir"]+img_name.replace(".nii.gz", "_L2_RAS_1.5_1.5_2.0.npy"))

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
        plt.savefig(train_dict["root_dir"]+"JN_{}.png".format(img_name), dpi=300)