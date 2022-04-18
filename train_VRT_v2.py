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

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "VRT_Iman_v3_008denoise"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 813
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["gpu_ids"] = [4]
train_dict["epochs"] = 100
train_dict["batch"] = 1
train_dict["dropout"] = 0
train_dict["model_term"] = "VRT"
train_dict["deconv_channels"] = 6
train_dict["input_size"] = [6,192,192]

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
train_dict["pre_train"] = "008_VRT_videodenoising_DAVIS.pth"
train_dict["spy_net"] = "spynet_sintel_final-3d2a1287.pth"
train_dict["val_ratio"] = 0.3
train_dict["test_ratio"] = 0.2

train_dict["loss_term"] = "L1_Charbonnier_loss_1e-9"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 4e-4 # default
train_dict["opt_betas"] = (0.9, 0.99) # default
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

# 008_train_vrt_videodenoising_davis
model = VRT(
    upscale=1, 
    img_size=[6,192,192], 
    window_size=[6,8,8], 
    depths=[8,8,8,8,8,8,8, 4,4, 4,4],
    indep_reconsts=[9,10], 
    embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], 
    pa_frames=2, 
    deformable_groups=16,
    nonblind_denoising=True
    )

pretrain = torch.load("./pre_train/"+train_dict["pre_train"])
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
criterion = L1_Charbonnier_loss()

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

# for idx_epoch in range(train_dict["epochs"]):

# pretrain_list = sorted(glob.glob(train_dict["save_folder"]+"*.pth"))
# pretrain_epoch = []
# for pretrain_path in pretrain_list:
#     print(pretrain_path, int(pretrain_path[-7:-4])-1)
#     idx_epoch = int(pretrain_path[-7:-4])-1

#     if idx_epoch < 18:
#         continue

#     model = torch.load(pretrain_path)
#     model.eval()
#     model = model.to(device)
#     criterion = nn.MSELoss()

    # print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

package_train = [train_list[:5], True, False, "train"]
package_val = [val_list[:5], False, True, "val"]
# package_test = [test_list, False, False, "test"]

for package in [package_val]:  # package_train 

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
    # N, D, C, H, W
    # x_data = nib.load(file_list[0]).get_fdata()

    for cnt_file, file_path in enumerate(file_list):
        
        x_path = file_path
        y_path = file_path.replace("MR", "CT")
        file_name = os.path.basename(file_path)
        print(iter_tag + " ===> Epoch[{:03d}]: --->".format(idx_epoch+1), x_path, "<---", end="")
        x_file = nib.load(x_path)
        y_file = nib.load(y_path)
        x_data = x_file.get_fdata()
        y_data = y_file.get_fdata()
        # x_data = x_data / np.amax(x_data)

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
                
                # print("H:", h_offset, h_offset+train_dict["input_size"][1])
                # print("W:", w_offset, w_offset+train_dict["input_size"][2])
                # print("Z:", z_center-1, z_center+2)

                batch_x[idx_batch, idx_channel, 0, :, :] = x_slice[:, :, 0]
                batch_x[idx_batch, idx_channel, 1, :, :] = x_slice[:, :, 1]
                batch_x[idx_batch, idx_channel, 2, :, :] = x_slice[:, :, 2]
                batch_y[idx_batch, idx_channel, 0, :, :] = y_slice[:, :, 0]
                batch_y[idx_batch, idx_channel, 1, :, :] = y_slice[:, :, 1]
                batch_y[idx_batch, idx_channel, 2, :, :] = y_slice[:, :, 2]

        batch_x = torch.from_numpy(batch_x).float().to(device)
        batch_y = torch.from_numpy(batch_y).float().to(device)

        # optimizer.zero_grad()
        y_hat = model(batch_x)
        # print("Yhat size: ", y_hat.size())
        loss = criterion(y_hat, batch_y)
        if isTrain:
            loss.backward()
            optimizer.step()
        case_loss[cnt_file] = loss.item()
        # print("Loss: ", case_loss[cnt_file])

    print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
    print("  Loss: ", np.mean(case_loss))
    np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

    if np.mean(case_loss) < best_val_loss:
        save the best model
        torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch+1))
        torch.save(optimizer, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
        print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
        best_val_loss = np.mean(case_loss)

        # if isVal:
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            # np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

            # if np.mean(case_loss) < best_val_loss:
            #     # save the best model
            #     torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch+1))
            #     print("Checkpoint saved at Epoch {:03d}".format(idx_epoch+1))
            #     best_val_loss = np.mean(case_loss)

        torch.cuda.empty_cache()
