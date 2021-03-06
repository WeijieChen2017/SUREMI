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

from model import SwinUNETR

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "SwinUNETR_Iman_v4_mse"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["input_size"] = [64, 64, 64]
train_dict["gpu_ids"] = [4]
train_dict["epochs"] = 100
train_dict["batch"] = 12
train_dict["case_iter_time"] = 5
train_dict["dropout"] = 0
train_dict["model_term"] = "SwinUNETR"

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
# train_dict["pre_train"] = "swin_base_patch244_window1677_kinetics400_22k.pth"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "MSELoss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

train_dict["model_related"] = {}
train_dict["model_related"]["in_channels"] = 1
train_dict["model_related"]["out_channels"] = 1
train_dict["model_related"]["feature_size"] = 60
train_dict["model_related"]["depths"] = [2, 2, 8, 2]
train_dict["model_related"]["num_heads"] = [3, 6, 12, 24]
train_dict["model_related"]["norm_name"] = "instance"
train_dict["model_related"]["drop_rate"] = 0.1
train_dict["model_related"]["attn_drop_rate"] = 0.1
train_dict["model_related"]["dropout_path_rate"] = 0.1
train_dict["model_related"]["use_checkpoint"] = True

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

print("="*60)
for k in train_dict:
    v = train_dict[k]
    if type(v) is dict:
        print("-->",k)
        for kk in v:
            print("-->-->", kk, v[kk], type(v[kk]))
    else:
        print("-->",k,v,type(v))
print("="*60)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SwinUNETR
model = SwinUNETR(
    img_size = train_dict["input_size"],
    in_channels = train_dict["model_related"]["in_channels"],
    out_channels = train_dict["model_related"]["out_channels"],
    feature_size = train_dict["model_related"]["feature_size"],
    depths = train_dict["model_related"]["depths"],
    num_heads = train_dict["model_related"]["num_heads"],
    norm_name = train_dict["model_related"]["norm_name"],
    drop_rate = train_dict["model_related"]["drop_rate"],
    attn_drop_rate = train_dict["model_related"]["attn_drop_rate"],
    dropout_path_rate = train_dict["model_related"]["dropout_path_rate"],
    use_checkpoint = train_dict["model_related"]["use_checkpoint"],
    )

# pretrain = torch.load("./pre_train/"+train_dict["pre_train"], map_location=torch.device('cpu'))
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
# model = torch.load(train_dict["save_folder"]+"model_best_093.pth", map_location=torch.device('cpu'))
# optimizer = torch.load(train_dict["save_folder"]+"optim_093.pth")

# model = nn.DataParallel(model)
model.train()
model = model.to(device)
criterion = nn.MSELoss()

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

# data_division_dict = np.load(train_dict["save_folder"]+"data_division.npy", allow_pickle=True).item()
# train_list = data_division_dict["train_list_X"]
# val_list = data_division_dict["val_list_X"]
# test_list = data_division_dict["test_list_X"]

# ==================== training ====================

best_val_loss = 1e6
best_epoch = 0
# wandb.watch(model)

package_train = [train_list, True, False, "train"] #[:10]l
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + 0
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]: # 

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
        cit_loss = np.zeros((train_dict["case_iter_time"]))

        # N, C, D, H, W
        x_data = nib.load(file_list[0]).get_fdata()
        cnt_total_file = len(file_list)

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            y_path = file_path.replace("MR", "CT")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]->[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, 
                cnt_file+1,
                cnt_total_file), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()
            # x_data = x_data / np.amax(x_data)

            acmu_grad = 0 # accumulate gradients 

            for idx_cit in range(train_dict["case_iter_time"]):

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
                
                if isVal:
                    with torch.no_grad():
                        y_hat = model(batch_x)
                        loss = criterion(y_hat, batch_y)
                if isTrain:
                    optimizer.zero_grad()
                    y_hat = model(batch_x)
                    loss = criterion(y_hat, batch_y)
                    loss.backward()
                    acmu_grad += 1

                    if acmu_grad == train_dict["case_iter_time"]-1:
                        optimizer.step()
                        acmu_grad = 0

                cit_loss[idx_cit] = loss.item()
            case_loss[cnt_file] = np.mean(cit_loss)
            print("Loss: ", case_loss[cnt_file])

        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        # if np.mean(case_loss) < best_val_loss:
        #     # save the best model
        #     torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
        #     print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
        #     best_val_loss = np.mean(case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

            # torch.save(model, train_dict["save_folder"]+"model_.pth".format(idx_epoch + 1))
            if np.mean(case_loss) < best_val_loss:
                # save the best model
                best_epoch = idx_epoch + 1
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optimizer, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = np.mean(case_loss)

        del batch_x, batch_y
        gc.collect()
        torch.cuda.empty_cache()
