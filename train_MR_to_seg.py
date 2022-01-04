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

from model import UNet, UNet_seg

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "MR_to_seg"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_channel"] = 3
train_dict["output_channel"] = 3
train_dict["gpu_ids"] = [5]
train_dict["epochs"] = 50
train_dict["batch"] = 10
train_dict["dropout"] = 0
train_dict["model_term"] = "UNet_seg"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/seg_CT/"
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



model = UNet_seg(n_channels=train_dict["input_channel"], n_classes=train_dict["output_channel"])
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

X_list = np.asarray(X_list)
np.random.shuffle(X_list)
X_list = list(X_list)

val_list_X = X_list[:int(len(X_list)*train_dict["val_ratio"])]
val_list_X.sort()
test_list_X = X_list[-int(len(X_list)*train_dict["test_ratio"]):]
test_list_X.sort()
train_list_X = list(set(X_list) - set(val_list_X) - set(test_list_X))
train_list_X.sort()

data_division_dict = {
    "train_list_X" : train_list_X,
    "val_list_X" : val_list_X,
    "test_list_X" : test_list_X}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)

# ==================== training ====================

best_val_loss = 1e6
wandb.watch(model)

for idx_epoch in range(train_dict["epochs"]):
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    package_train = [train_list_X, True, False, "train"]
    package_val = [val_list_X, False, True, "val"]
    package_test = [test_list_X, False, False, "test"]

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
            
            cube_x_path = file_path
            file_name = os.path.basename(file_path)
            cube_y1_path = train_dict["folder_Y"] + "/air/" + file_name
            cube_y2_path = train_dict["folder_Y"] + "/bone/" + file_name
            cube_y3_path = train_dict["folder_Y"] + "/soft_tissue/" + file_name
            print("--->",cube_x_path,"<---", end="")
            cube_x_data = nib.load(cube_x_path).get_fdata()
            cube_y1_data = nib.load(cube_y1_path).get_fdata()
            cube_y2_data = nib.load(cube_y2_path).get_fdata()
            cube_y3_data = nib.load(cube_y3_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss = np.zeros((len_z//train_dict["batch"]))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            for idx_iter in range(len_z//train_dict["batch"]):

                batch_x = np.zeros((train_dict["batch"], train_dict["input_channel"], cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((train_dict["batch"], train_dict["output_channel"], cube_y1_data.shape[0], cube_y1_data.shape[1]))

                for idx_batch in range(train_dict["batch"]):
                    z_center = input_list[idx_iter*train_dict["batch"]+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1

                    if train_dict["input_channel"] == 3:
                        batch_x[idx_batch, 1, :, :] = cube_x_data[:, :, z_center]
                        batch_x[idx_batch, 0, :, :] = cube_x_data[:, :, z_before]
                        batch_x[idx_batch, 2, :, :] = cube_x_data[:, :, z_after]
                    if train_dict["input_channel"] == 1:
                        batch_x[idx_batch, 0, :, :] = cube_x_data[:, :, z_center]
                    if train_dict["output_channel"] == 3:
                        batch_y[idx_batch, 0, :, :] = cube_y1_data[:, :, z_center]
                        batch_y[idx_batch, 1, :, :] = cube_y2_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = cube_y3_data[:, :, z_center]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)

                optimizer.zero_grad()
                y_hat = model(batch_x)
                loss = criterion(y_hat, batch_y)
                if isTrain:
                    loss.backward()
                    optimizer.step()

                case_loss[idx_iter] = loss.item()
                case_loss[idx_iter] = loss.item()
            
            case_name = os.path.basename(cube_x_path)[5:8]
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
