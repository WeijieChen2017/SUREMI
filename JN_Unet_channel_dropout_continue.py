import os
import torch
# from model import UNet_Theseus as UNet
from model import UNet_channelDO as UNet
from monai.networks.layers.factories import Act, Norm

# from model import UNETR_bdo as UNETR

model_hub = [
    ["Seg532_Unet_channnel_r100" , 0.50, 1, [6], False], # 22GB 
    ["Seg532_Unet_channnel_r100w", 0.50, 1, [7], True, 21000],  # 22GB # to recover this
    ["Seg532_Unet_channnel_r050", 0.50, 0.5, [0], False],  # 22GB 
    ["Seg532_Unet_channnel_r050w", 0.50, 0.5, [0], True],  # 22GB 
]

print("Model index: ", end="")
model_idx = int(input()) - 1


model_name = model_hub[model_idx][0]
dropout_ratio = model_hub[model_idx][1]
reg_coef = model_hub[model_idx][2]
GPU_available = model_hub[model_idx][3]
is_WDO = model_hub[model_idx][4]
global_step = model_hub[model_idx][5]


train_dict = {}
train_dict["root_dir"] = "./project_dir/"+model_name+"/"
if not os.path.exists(train_dict["root_dir"]):
    os.mkdir(train_dict["root_dir"])
train_dict["data_dir"] = "./data_dir/JN_BTCV/"
train_dict["split_JSON"] = "dataset_532.json"
train_dict["gpu_list"] = GPU_available

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
# from monai.networks.nets.unet import UNet

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)



print_config()

#--------------------------------------------------------------
# print("Press any key to continue:", end="")
# _ = input()
#--------------------------------------------------------------

# directory = os.environ.get("./project_dir/JN_UnetR/")
# root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = train_dict["root_dir"]
print(root_dir)

#--------------------------------------------------------------
# print("Press any key to continue:", end="")
# _ = input()
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
# print("Press any key to continue:", end="")
# _ = input()
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
# print("Press any key to continue:", end="")
# _ = input()
#--------------------------------------------------------------

gpu_list = ','.join(str(x) for x in train_dict["gpu_list"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = UNet( 
    spatial_dims=3,
    in_channels=1,
    out_channels=14,
    channels=(64, 128, 256, 512),
    strides=(2, 2, 2),
    num_res_units=6,
    act=Act.PRELU,
    norm=Norm.INSTANCE,
    dropout=dropout_ratio,
    bias=True,
    is_WDO=is_WDO,
    ).to(device)


pre_state = torch.load(train_dict["root_dir"]+"UNETR_model_best_acc.pth", map_location=torch.device('cpu'))
# pre_state = model_pre.state_dict()
model_state_keys = model.state_dict().keys()
new_model_state = {}

for model_key in model_state_keys:
    print(model_key)
    new_model_state[model_key] = pre_state[model_key]
    
model.load_state_dict(new_model_state)
print("./pre_train/UNETR_model_best_acc.pth has been loaded!")



# loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
loss_function_rec = DiceCELoss(to_onehot_y=True, softmax=True)
loss_function_reg = torch.nn.MSELoss()
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

#--------------------------------------------------------------
# print("Press any key to continue:", end="")
# _ = input()
#--------------------------------------------------------------

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            # np.save("val_inputs.npy", val_inputs.cpu().detach().numpy())
            # np.save("val_labels.npy", val_labels.cpu().detach().numpy())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, model)
            # np.save("val_outputs.npy", val_outputs.cpu().detach().numpy())
            val_labels_list = decollate_batch(val_labels)
            # np.save("val_labels_list.npy", val_labels_list.cpu().detach().numpy())
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            # np.save("val_labels_convert.npy", val_labels_convert.cpu().detach().numpy())
            val_outputs_list = decollate_batch(val_outputs)
            # np.save("val_outputs_list.npy", val_outputs_list.cpu().detach().numpy())
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            # np.save("val_output_convert.npy", val_output_convert.cpu().detach().numpy())
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
        loss = loss_function_rec(logit_map, y)+loss_function_reg(logit_map, logit_map_reg) * reg_coef
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
# global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))

#--------------------------------------------------------------
# print("Press any key to continue:", end="")
# _ = input()
#--------------------------------------------------------------

print(
    f"train completed, best_metric: {dice_val_best:.4f} "
    f"at iteration: {global_step_best}"
)
