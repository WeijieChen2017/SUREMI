import os
import glob
import time
import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

import torch
import torchvision
import requests

from model import SwinTransformer3D

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "Swin3d_to_CT"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_channel"] = 1
train_dict["output_channel"] = 1
train_dict["gpu_ids"] = [7]
train_dict["epochs"] = 50
train_dict["batch"] = 4
train_dict["dropout"] = 0
train_dict["model_term"] = "SwinTransformer3D"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "SmoothL1Loss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

wandb.init(project=train_dict["project_name"])
config = wandb.config
config.in_chan = train_dict["input_channel"]
config.out_chan = train_dict["output_channel"]
config.epochs = train_dict["epochs"]
config.batch = train_dict["batch"]
config.dropout = train_dict["dropout"]
config.moodel_term = train_dict["model_term"]
config.loss_term = train_dict["loss_term"]
config.opt_lr = train_dict["opt_lr"]
config.opt_weight_decay = train_dict["opt_weight_decay"]

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

depths=[2, 2, 18, 2],
embed_dim=128,
num_heads=[4, 8, 16, 32]

# Swin-B
model = SwinTransformer3D(
    pretrained=None,
    pretrained2d=True,
    patch_size=(2,4,4),
    in_chans=3,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=(16,7,7),
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.,
    attn_drop_rate=0.,
    drop_path_rate=0.2,
    norm_layer=nn.LayerNorm,
    patch_norm=False,
    frozen_stages=-1,
    use_checkpoint=False)


model_state_dict_MR = model.state_dict()
model_state_dict_CT = model.state_dict()
model_state_dict_MR.update(new_state_dict_MR)
model_state_dict_CT.update(new_state_dict_CT)
model.load_state_dict(model_state_dict_MR)
model.load_state_dict(model_state_dict_CT)

model_state_dict = model.state_dict()
dict_name = list(model_state_dict)
for i, p in enumerate(dict_name):
    print(i, p)



model.train().float()
model = model.to(device)
criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

selected_list = np.asarray(Y_list)
np.random.shuffle(selected_list)
selected_list = list(selected_list)

val_list = selected_list[:int(len(selected_list)*train_dict["val_ratio"])]
val_list.sort()
test_list = selected_list[-int(len(selected_list)*train_dict["test_ratio"]):]
test_list.sort()
train_list = list(set(selected_list) - set(val_list) - set(test_list))
train_list.sort()

data_division_dict = {
    "train_list_X" : train_list,
    "val_list_X" : val_list,
    "test_list_X" : test_list}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e6
wandb.watch(model)

for idx_epoch in range(train_dict["epochs"]):
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    package_train = [train_list, True, False, "train"]
    package_val = [val_list, False, True, "val"]
    package_test = [test_list, False, False, "test"]

    for package in [package_train, package_val, package_test]:

        file_list = package[0]
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            model.train()
        else:
            model.eval()

        random.shuffle(file_list)
        epoch_loss = np.zeros((len(file_list)))
        for cnt_file, file_path in enumerate(file_list):
            
            cube_y_path = file_path
            file_name = os.path.basename(file_path)
            cube_x1_path = train_dict["folder_X"] + "/air/" + file_name
            cube_x2_path = train_dict["folder_X"] + "/bone/" + file_name
            cube_x3_path = train_dict["folder_X"] + "/soft_tissue/" + file_name
            print("--->",cube_y_path,"<---", end="")
            cube_y_data = nib.load(cube_y_path).get_fdata()
            cube_x1_data = nib.load(cube_x1_path).get_fdata()
            cube_x2_data = nib.load(cube_x2_path).get_fdata()
            cube_x3_data = nib.load(cube_x3_path).get_fdata()
            len_z = cube_y_data.shape[2]
            case_loss = np.zeros((len_z//train_dict["batch"]))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            for idx_iter in range(len_z//train_dict["batch"]):

                batch_x = np.zeros((train_dict["batch"], train_dict["input_channel"], cube_x1_data.shape[0], cube_x1_data.shape[1]))
                batch_y = np.zeros((train_dict["batch"], train_dict["output_channel"], cube_y_data.shape[0], cube_y_data.shape[1]))

                for idx_batch in range(train_dict["batch"]):
                    z_center = input_list[idx_iter*train_dict["batch"]+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1

                    if train_dict["input_channel"] == 3:
                        batch_x[idx_batch, 0, :, :] = cube_x1_data[:, :, z_center]
                        batch_x[idx_batch, 1, :, :] = cube_x2_data[:, :, z_center]
                        batch_x[idx_batch, 2, :, :] = cube_x3_data[:, :, z_center]

                    if train_dict["output_channel"] == 3:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_before]
                        batch_y[idx_batch, 1, :, :] = cube_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = cube_y_data[:, :, z_after]
                    if train_dict["output_channel"] == 1:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_center]

                batch_x = 
                batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)

                optimizer.zero_grad()
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                if isTrain:
                    loss.backward()
                    optimizer.step()

                case_loss[idx_iter] = loss.item()
                case_loss[idx_iter] = loss.item()
            
            case_name = os.path.basename(cube_y_path)[5:8]
            if not isTrain:
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_z.npy".format(idx_epoch+1, case_name), y_hat.cpu().detach().numpy())

            # after training one case
            loss_mean = np.mean(case_loss)
            loss_std = np.std(case_loss)
            print("===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt_file+1), end='')
            print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
            epoch_loss[cnt_file] = loss_mean

        # after training all cases
        loss_mean = np.mean(epoch_loss)
        loss_std = np.std(epoch_loss)
        print("===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss mean: {:.6} Loss std: {:.6}".format(loss_mean, loss_std))
        wandb.log({iter_tag+"_loss": loss_mean})
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), epoch_loss)

        if isVal:
            if loss_mean < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch+1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
                best_val_loss = loss_mean

        torch.cuda.empty_cache()
