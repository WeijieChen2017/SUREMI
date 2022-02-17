import os
import gc
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn

import torch
import torchvision
import requests

from model import VRT

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "Swin3d_to_CT"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["gpu_ids"] = [1]
train_dict["epochs"] = 50
train_dict["batch"] = 2
train_dict["dropout"] = 0
train_dict["model_term"] = "VRT"
train_dict["deconv_channels"] = 6
train_dict["input_size"] = [6,64,64]

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"
train_dict["pre_train"] = "007_VRT_videodeblurring_REDS.pth"
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

# wandb.init(project=train_dict["project_name"])
# config = wandb.config
# config.in_chan = train_dict["input_channel"]
# config.out_chan = train_dict["output_channel"]
# config.epochs = train_dict["epochs"]
# config.batch = train_dict["batch"]
# config.dropout = train_dict["dropout"]
# config.moodel_term = train_dict["model_term"]
# config.loss_term = train_dict["loss_term"]
# config.opt_lr = train_dict["opt_lr"]
# config.opt_weight_decay = train_dict["opt_weight_decay"]

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# VRT-007-3motion-blur
model = VRT(
    upscale=1, 
    img_size=[6,192,192], 
    window_size=[6,8,8], 
    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
    indep_reconsts=[9,10], 
    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], 
    pa_frames=2, 
    deformable_groups=16
    )

pretrain = torch.load("./pre_train/"+train_dict["pre_train"], map_location=torch.device('cpu'))
model.load_state_dict(pretrain["params"])
# pretrain_state = pretrain["state_dict"]
# pretrain_state_keys = pretrain_state.keys()
# model_state_keys = model.state_dict().keys()
# new_model_state = {}

# del pretrain
# gc.collect()
# torch.cuda.empty_cache()

# for model_key in model_state_keys:
#     if "backbone."+model_key in pretrain_state_keys:
#         new_model_state[model_key] = pretrain_state["backbone."+model_key]
#     else:
#         new_model_state[model_key] = model.state_dict()[model_key]

# model.load_state_dict(new_model_state)

# model = nn.DataParallel(model)
model.train()
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

selected_list = np.asarray(X_list)
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
# wandb.watch(model)

for idx_epoch in range(train_dict["epochs"]):
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    package_train = [train_list, True, False, "train"]
    package_val = [val_list, False, True, "val"]
    # package_test = [test_list, False, False, "test"]

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
        
        case_loss = np.zeros((len(file_list)))

        # b, d, c, h, w
        x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            
            x_path = file_path
            y_path = file_path.replace("MR", "CT")
            file_name = os.path.basename(file_path)
            print("===> Epoch[{:03d}]: --->".format(idx_epoch+1), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()

            for idx_batch in range(train_dict["batch"]):

                batch_x = np.zeros((train_dict["batch"], train_dict["input_size"][0], 3, train_dict["input_size"][1], train_dict["input_size"][2]))
                batch_y = np.zeros((train_dict["batch"], train_dict["input_size"][0], 3, train_dict["input_size"][1], train_dict["input_size"][2]))

                z_offset = np.random.randint(x_data.shape[2]//3 - train_dict["input_size"][0])
                h_offset = np.random.randint(x_data.shape[0] - train_dict["input_size"][1])
                w_offset = np.random.randint(x_data.shape[1] - train_dict["input_size"][2])

                for idx_channel in range(train_dict["input_size"][0]):
                    z_center = (z_offset + idx_channel) * 3 + 1
                    x_slice = x_data[h_offset:h_offset+train_dict["input_size"][1], w_offset:w_offset+train_dict["input_size"][2], z_center-1:z_center+2]
                    y_slice = y_data[h_offset:h_offset+train_dict["input_size"][1], w_offset:w_offset+train_dict["input_size"][2], z_center-1:z_center+2]
                    
                    batch_x[idx_batch, idx_channel, 0, :, :] = x_slice[:, :, 0]
                    batch_x[idx_batch, idx_channel, 1, :, :] = x_slice[:, :, 1]
                    batch_x[idx_batch, idx_channel, 2, :, :] = x_slice[:, :, 2]
                    batch_y[idx_batch, idx_channel, 0, :, :] = y_slice[:, :, 0]
                    batch_y[idx_batch, idx_channel, 1, :, :] = y_slice[:, :, 1]
                    batch_y[idx_batch, idx_channel, 2, :, :] = y_slice[:, :, 2]

            batch_x = torch.from_numpy(batch_x).float().to(device)
            batch_y = torch.from_numpy(batch_y).float().to(device)

            optimizer.zero_grad()
            y_hat = model(batch_x)
            # print("Yhat size: ", y_hat.size())
            loss = criterion(y_hat, batch_y)
            if isTrain:
                loss.backward()
                optimizer.step()
            case_loss[cnt_file] = loss.item()
            print("Loss: ", case_loss[cnt_file])

        print("===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_x.npy".format(idx_epoch+1, file_name), batch_x.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_y.npy".format(idx_epoch+1, file_name), batch_y.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_z.npy".format(idx_epoch+1, file_name), y_hat.cpu().detach().numpy())

            if np.mean(epoch_loss) < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch+1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
                best_val_loss = np.mean(epoch_loss)

        torch.cuda.empty_cache()
