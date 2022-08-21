from model import UNETR_bdo as UNETR

train_dict = {}
train_dict["state_weight_mapping_1"] = swm_1
train_dict["state_weight_mapping_2"] = swm_1
train_dict["target_model_1"] = "./project_dir/Unet_Monai_Iman_v3/model_best_047.pth"
train_dict["target_model_2"] = "./project_dir/Unet_Monai_Iman_v3/model_best_057.pth"
train_dict["root_dir"] = "./project_dir/JN_UnetR_bdo/"
train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_0.json"
train_dict["gpu_list"] = [6,7]

import os
import gc
import copy
import glob
import time
import shutil
import random
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

print_config()

#--------------------------------------------------------------
print("Press any key to continue:", end="")
_ = input()
#--------------------------------------------------------------

# directory = os.environ.get("./project_dir/JN_UnetR/")
# root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = train_dict["root_dir"]
print(root_dir)

#--------------------------------------------------------------
print("Press any key to continue:", end="")
_ = input()
#--------------------------------------------------------------

train_transforms = Compose(
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
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
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

#--------------------------------------------------------------
print("Press any key to continue:", end="")
_ = input()
#--------------------------------------------------------------

data_dir = train_dict["data_dir"]
split_JSON = train_dict["split_JSON"]

datasets = data_dir + split_JSON
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
)
val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)

#--------------------------------------------------------------
print("Press any key to continue:", end="")
_ = input()
#--------------------------------------------------------------

gpu_list = ','.join(str(x) for x in train_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# state weights mapping
swm_1 = {}
swm_1["vit.0"]      = "vit"
swm_1["encoder1.0"] = "encoder1"
swm_1["encoder2.0"] = "encoder2"
swm_1["encoder3.0"] = "encoder3"
swm_1["encoder4.0"] = "encoder4"
swm_1["decoder5.0"] = "decoder5"
swm_1["decoder4.0"] = "decoder4"
swm_1["decoder3.0"] = "decoder3"
swm_1["decoder2.0"] = "decoder2"
swm_1["out.0"] = "out"

swm_2 = {}
swm_2["vit.1"]      = "vit"
swm_2["encoder1.1"] = "encoder1"
swm_2["encoder2.1"] = "encoder2"
swm_2["encoder3.1"] = "encoder3"
swm_2["encoder4.1"] = "encoder4"
swm_2["decoder5.1"] = "decoder5"
swm_2["decoder4.1"] = "decoder4"
swm_2["decoder3.1"] = "decoder3"
swm_2["decoder2.1"] = "decoder2"
swm_2["out.1"] = "out"
train_dict["state_weight_mapping_1"] = swm_1
train_dict["state_weight_mapping_2"] = swm_1
train_dict["target_model_1"] = "./project_dir/JN_UnetR/best_metric_model.pth"
train_dict["target_model_2"] = "./project_dir/JN_UnetR/best_metric_model.pth"

pretrain_1 = torch.load(train_dict["target_model_1"])
pretrain_1_state = pretrain_1.state_dict()
pretrain_2 = torch.load(train_dict["target_model_2"])
pretrain_2_state = pretrain_2.state_dict()

model_state_keys = model.state_dict().keys()
new_model_state = {}

for model_key in model_state_keys:
    weight_prefix = model_key[:model_key.find(".")+2]

    # in the first half
    if weight_prefix in swm_1.keys():
        weight_replacement = swm_1[weight_prefix]
        new_model_state[model_key] = pretrain_1_state[model_key.replace(weight_prefix, weight_replacement)]

    # in the second half
    if weight_prefix in swm_2.keys():
        weight_replacement = swm_2[weight_prefix]
        new_model_state[model_key] = pretrain_2_state[model_key.replace(weight_prefix, weight_replacement)]
    
model.load_state_dict(new_model_state)


loss_function_rec = DiceCELoss(to_onehot_y=True, softmax=True)
loss_function_reg = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

#--------------------------------------------------------------
print("Press any key to continue:", end="")
_ = input()
#--------------------------------------------------------------

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        logit_map = model(x)
        logit_map_reg = model(x)
        loss = loss_function_rec(logit_map, y)+loss_function_reg(logit_map, logit_map_reg)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )
        np.save(train_dict["root_dir"]+"step_loss_train_{:03d}.npy".format(step+1), loss.item())
        if (
            global_step % eval_num == 0 and global_step != 0
        ) or global_step == max_iterations:
            epoch_iterator_val = tqdm(
                val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
            )
            dice_val = validation(epoch_iterator_val)
            np.save(train_dict["root_dir"]+"step_loss_cal_{:03d}.npy".format(step+1), dice_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(
                    model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
                )
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
# model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

#--------------------------------------------------------------
# print("Press any key to continue:", end="")
# _ = input()
#--------------------------------------------------------------

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)

model_list = [
    ["Theseus_v2_47_57_rdp000", [5], 0.,],
    ["Theseus_v2_47_57_rdp020", [5], 0.2,],
    ["Theseus_v2_47_57_rdp040", [6], 0.4,],
    ["Theseus_v2_47_57_rdp060", [6], 0.6,],
    ["Theseus_v2_47_57_rdp080", [7], 0.8,],
    ["Theseus_v2_47_57_rdp100", [7], 1.,],
    ]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)
# current_model_idx = 0
# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = model_list[current_model_idx][0]
train_dict["gpu_ids"] = model_list[current_model_idx][1]
train_dict["alpha_dropout_consistency"] = model_list[current_model_idx][2]

train_dict["dropout"] = 0.
train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [96, 96, 96]
train_dict["epochs"] = 200
train_dict["batch"] = 16
train_dict["well_trained_model"] = "./project_dir/Unet_Monai_Iman_v2/model_best_181.pth"

train_dict["model_term"] = "Monai_Unet_MacroDropout"
train_dict["dataset_ratio"] = 1
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False


unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["out_channels"] = 1
unet_dict["channels"] = (32, 64, 128, 256)
unet_dict["strides"] = (2, 2, 2)
unet_dict["num_res_units"] = 4
unet_dict["act"] = Act.PRELU
unet_dict["normunet"] = Norm.INSTANCE
unet_dict["dropout"] = train_dict["dropout"]
unet_dict["bias"] = True
train_dict["model_para"] = unet_dict




# state weights mapping
swm_1 = {}
swm_1["down1.0"]   = "model.0"
swm_1["down2.0"]   = "model.1.submodule.0"
swm_1["down3.0"]   = "model.1.submodule.1.submodule.0"
swm_1["bottom.0"]  = "model.1.submodule.1.submodule.1.submodule"
swm_1["up3.0"]     = "model.1.submodule.1.submodule.2"
swm_1["up2.0"]     = "model.1.submodule.2"
swm_1["up1.0"]     = "model.2"

swm_2 = {}
swm_1["down1.1"]   = "model.0"
swm_1["down2.1"]   = "model.1.submodule.0"
swm_1["down3.1"]   = "model.1.submodule.1.submodule.0"
swm_1["bottom.1"]  = "model.1.submodule.1.submodule.1.submodule"
swm_1["up3.1"]     = "model.1.submodule.1.submodule.2"
swm_1["up2.1"]     = "model.1.submodule.2"
swm_1["up1.1"]     = "model.2"
train_dict["state_weight_mapping_1"] = swm_1
train_dict["state_weight_mapping_2"] = swm_1
train_dict["target_model_1"] = "./project_dir/Unet_Monai_Iman_v3/model_best_047.pth"
train_dict["target_model_2"] = "./project_dir/Unet_Monai_Iman_v3/model_best_057.pth"


train_dict["folder_X"] = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"
train_dict["folder_Y"] = "./project_dir/Unet_Monai_Iman_v2/pred_monai/"
# train_dict["pre_train"] = "swin_base_patch244_window1677_kinetics400_22k.pth"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = UNet( 
    spatial_dims=unet_dict["spatial_dims"],
    in_channels=unet_dict["in_channels"],
    out_channels=unet_dict["out_channels"],
    channels=unet_dict["channels"],
    strides=unet_dict["strides"],
    num_res_units=unet_dict["num_res_units"],
    act=unet_dict["act"],
    norm=unet_dict["normunet"],
    dropout=unet_dict["dropout"],
    bias=unet_dict["bias"],
    )

pretrain_1 = torch.load(train_dict["target_model_1"])
pretrain_1_state = pretrain_1.state_dict()
pretrain_2 = torch.load(train_dict["target_model_2"])
pretrain_2_state = pretrain_2.state_dict()

model_state_keys = model.state_dict().keys()
new_model_state = {}

for model_key in model_state_keys:
    weight_prefix = model_key[:model_key.find(".")+2]

    # in the first half
    if weight_prefix in swm_1.keys():
        weight_replacement = swm_1[weight_prefix]
        new_model_state[model_key] = pretrain_1_state[model_key.replace(weight_prefix, weight_replacement)]

    # in the second half
    if weight_prefix in swm_2.keys():
        weight_replacement = swm_2[weight_prefix]
        new_model_state[model_key] = pretrain_2_state[model_key.replace(weight_prefix, weight_replacement)]
    
model.load_state_dict(new_model_state)

model.train()
model = model.to(device)

# optim = torch.optim.RMSprop(model.parameters(), lr=train_dict["opt_lr"])
loss_fnc = torch.nn.SmoothL1Loss()
loss_doc = torch.nn.SmoothL1Loss()
# loss_fnc = torch.nn.CrossEntropyLoss()

optim = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

train_list = glob.glob(train_dict["folder_X"]+"*_xtr.nii.gz")
val_list = glob.glob(train_dict["folder_X"]+"*_xva.nii.gz")
test_list = glob.glob(train_dict["folder_X"]+"*_xte.nii.gz")

data_division_dict = {
    "train_list_X" : train_list,
    "val_list_X" : val_list,
    "test_list_X" : test_list}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0

package_train = [train_list, True, False, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + train_dict["continue_training_epoch"]
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]:

        file_list = package[0]
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model.train()
        else:
            model.eval()

        random.shuffle(file_list)
        
        case_loss = np.zeros((len(file_list), 2))

        # N, C, D, H, W
        x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            y_path = file_path.replace("x", "y")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, cnt_file+1, len(file_list)), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()

            batch_x = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
            batch_y = np.zeros((train_dict["batch"], 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

            for idx_batch in range(train_dict["batch"]):
                
                d0_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][1])
                d1_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][2])
                d2_offset = np.random.randint(x_data.shape[2] - train_dict["input_size"][0])

                x_slice = x_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                y_slice = y_data[d0_offset:d0_offset+train_dict["input_size"][0],
                                 d1_offset:d1_offset+train_dict["input_size"][1],
                                 d2_offset:d2_offset+train_dict["input_size"][2]
                                 ]
                batch_x[idx_batch, 0, :, :, :] = x_slice
                batch_y[idx_batch, 0, :, :, :] = y_slice

            batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)
            
            if isTrain:

                optim.zero_grad()
                y_hat = model(batch_x)
                y_ref = model(batch_x)
                loss_recon = loss_fnc(y_hat, batch_y)
                loss_rdrop = loss_doc(y_ref, y_hat)
                loss = loss_recon + loss_rdrop * train_dict["alpha_dropout_consistency"]
                loss.backward()
                optim.step()
                case_loss[cnt_file, 0] = loss_recon.item()
                case_loss[cnt_file, 1] = loss_rdrop.item()
                print("Loss: ", np.mean(case_loss[cnt_file, :]), "Recon: ", loss_recon.item(), "Rdropout: ", loss_rdrop.item())

            if isVal:

                with torch.no_grad():
                    y_hat = model(batch_x)
                    y_ref = model(batch_x)
                    loss_recon = loss_fnc(y_hat, batch_y)
                    loss_rdrop = loss_doc(y_ref, y_hat)
                    loss = loss_recon + loss_rdrop * train_dict["alpha_dropout_consistency"]

                case_loss[cnt_file, 0] = loss_recon.item()
                case_loss[cnt_file, 1] = loss_rdrop.item()
                print("Loss: ", np.mean(case_loss[cnt_file, :]), "Recon: ", loss_recon.item(), "Rdropout: ", loss_rdrop.item())

        epoch_loss_recon = np.mean(case_loss[:, 0])
        epoch_loss_rdrop = np.mean(case_loss[:, 1])
        # epoch_loss = np.mean(case_loss)
        epoch_loss = epoch_loss_recon
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("Loss: ", epoch_loss, "Recon: ", epoch_loss_recon, "Rdropout: ", epoch_loss_rdrop)
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_xf.npy", batch_xf.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_fmap.npy", batch_fmap.cpu().detach().numpy())
            torch.save(model, train_dict["save_folder"]+"model_curr.pth".format(idx_epoch + 1))
            
            if epoch_loss < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optim, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = epoch_loss

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()
