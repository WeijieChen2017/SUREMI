
# Generative Confidential Network

import os
import gc
import copy
import glob
import time
# import wandb
import random

import numpy as np
import nibabel as nib
import torch.nn as nn
import torch.nn.functional as F

import torch
import torchvision
import requests

from monai.networks.nets.unet import UNet as UNet
from monai.networks.layers.factories import Act, Norm
import bnn

from utils import add_noise, weighted_L1Loss
from model import GCN
# from torchsummary import summary

# v1 Gaussian mu=0, sigma=0.5
# v2 Gaussian mu=0, sigma=0.25
# v3 Poisson lambda=1
# v4 Poisson lambda=0.25
# v5 Salt&Pepper Salt=0.975, Pepper=0.025
# v6 Salt&Pepper Salt=0.95, Pepper=0.05
# v7 Speckle mu=0, sigma=0.25
# v8 Speckle mu=0, sigma=0.5
# v9 Racian snr=5
# v10 Racian snr=10
# v11 Rayleigh snr=5
# v12 Rayleigh snr=10

model_list = [
    # ["CM_HighDropout", "None", (0, ), "CT", [7], 0.5, 11],
    ["GCN_v3", "None", (0, ), "CT", [7], 0., 1, 0],
    # ["v1_Gau050_MRMR_dual", "Gaussian", (0, 0.5), "MR", [7]],
    # ["v1_Gau050_MRCT", "Gaussian", (0, 0.5), "CT", [7]],
    # ["v2_Gau025_MRMR", "Gaussian", (0, 0.25), "MR", [7]],
    # ["v2_Gau025_MRCT", "Gaussian", (0, 0.25), "CT", [7]],
    # ["v3_Poi100_MRMR", "Poisson", (1,), "MR", [6]],
    # ["v3_Poi100_MRCT", "Poisson", (1,), "CT", [6]],
    # ["v4_Poi025_MRMR", "Poisson", (0.25,), "MR", [6]],
    # ["v4_Poi025_MRCT", "Poisson", (0.25,), "CT", [6]],
    # ["v5_S&P025_MRMR", "Salt&Pepper", (0.975, 0.025), "MR", [3]],
    # ["v5_S&P025_MRCT", "Salt&Pepper", (0.975, 0.025), "CT", [3]],
    # ["v6_S&P050_MRMR", "Salt&Pepper", (0.95, 0.05), "MR", [3]],
    # ["v6_S&P050_MRCT", "Salt&Pepper", (0.95, 0.25), "CT", [3]],
    # ["v7_SPK025_MRMR", "Speckle", (0, 0.25), "MR", [7]],
    # ["v7_SPK025_MRCT", "Speckle", (0, 0.25), "CT", [7]],
    # ["v8_SPK050_MRMR", "Speckle", (0, 0.5), "MR", [6]],
    # ["v8_SPK050_MRCT", "Speckle", (0, 0.5), "CT", [6]],
    # ["v9_RIC005_MRMR", "Racian", (5,), "MR", [3]],
    # ["v9_RIC005_MRCT", "Racian", (5,), "CT", [3]],
    # ["v10_RIC010_MRMR", "Racian", (10,), "MR", [3]],
    # ["v10_RIC010_MRCT", "Racian", (10,), "CT", [3]],
    # ["v11_RAY005_MRMR", "Rayleigh", (5, ), "MR", [3]],
    # ["v11_RAY005_MRCT", "Rayleigh", (5, ), "CT", [3]],
    # ["v12_RAY010_MRMR", "Rayleigh", (10, ), "MR", [6]],
    # ["v12_RAY010_MRCT", "Rayleigh", (10, ), "CT", [6]],
    ]

print("Model index: ", end="")
current_model_idx = int(input()) - 1
print(model_list[current_model_idx])
time.sleep(1)
# current_model_idx = 0
# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "Bayesian_HDMGD_"+model_list[current_model_idx][0]
train_dict["noise_type"] = model_list[current_model_idx][1]
train_dict["noise_params"] = model_list[current_model_idx][2]
train_dict["target_img"] = model_list[current_model_idx][3]
train_dict["gpu_ids"] = model_list[current_model_idx][4]
train_dict["dropout"] = model_list[current_model_idx][5]
train_dict["n_MTGD"] = model_list[current_model_idx][6]
train_dict["alpha_loss_CM"] = model_list[current_model_idx][7]

train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [96, 96, 96]
train_dict["epochs"] = 200
train_dict["batch"] = 32
train_dict["well_trained_model"] = "./project_dir/Bayesian_HDMGD_GCN_v2/model_best_051.pth"

train_dict["beta"] = 1e6 # resize KL loss
train_dict["model_term"] = "Monai_Unet3d"
train_dict["dataset_ratio"] = 1
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False


unet_dict_E = {}
unet_dict_E["spatial_dims"] = 3
unet_dict_E["in_channels"] = 2
unet_dict_E["out_channels"] = 1
unet_dict_E["channels"] = (32, 64, 128, 256)
unet_dict_E["strides"] = (2, 2, 2)
unet_dict_E["num_res_units"] = 4
unet_dict_E["act"] = Act.PRELU
unet_dict_E["normunet"] = Norm.INSTANCE
unet_dict_E["dropout"] = 0.0
unet_dict_E["bias"] = True
train_dict["model_E"] = unet_dict_E


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

# model = GCN(unet_dict_G, unet_dict_E)

model_E = UNet( 
    spatial_dims=unet_dict_E["spatial_dims"],
    in_channels=unet_dict_E["in_channels"],
    out_channels=unet_dict_E["out_channels"],
    channels=unet_dict_E["channels"],
    strides=unet_dict_E["strides"],
    num_res_units=unet_dict_E["num_res_units"],
    act=unet_dict_E["act"],
    norm=unet_dict_E["normunet"],
    dropout=unet_dict_E["dropout"],
    bias=unet_dict_E["bias"],
    )

model_E.train()
model_E = model_E.to(device)

model_G = torch.load(train_dict["well_trained_model"], map_location=torch.device('cpu'))
model_G.eval()
model_G.to(device)

# criterion = nn.SmoothL1Loss()
# loss_L1 = weighted_L1Loss
# loss_L1 = nn.SmoothL1Loss()
# loss_L2 = nn.MSELoss()

optim = torch.optim.AdamW(
    model_E.parameters(),
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
len_dataset = len(selected_list)
selected_list = selected_list[:int(len_dataset * train_dict["dataset_ratio"])]

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


# ==================== MTGD ====================

# if train_dict["n_MTGD"] > 1:
#     MTGD_dict = {}
#     for model_key, param in model.named_parameters():
#         print(model_key)
#         new_shape = (train_dict["n_MTGD"], torch.flatten(param).size()[0])
#         MTGD_dict[model_key] = np.zeros(new_shape)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0
ONE_CM = torch.ones((
    train_dict["batch"],
    1, 
    train_dict["input_size"][0],
    train_dict["input_size"][1],
    train_dict["input_size"][2])).float().to(device)
# wandb.watch(model)

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
            model_E.train()
        else:
            model_E.eval()

        random.shuffle(file_list)
        
        case_loss = np.zeros((len(file_list), 3)), # wL1, CM, recon L1

        # N, C, D, H, W
        x_data = nib.load(file_list[0]).get_fdata()

        for cnt_file, file_path in enumerate(file_list):
            
            x_path = file_path
            y_path = file_path.replace("MR", "CT")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, cnt_file+1, len(file_list)), x_path, "<---", end="")
            x_file = nib.load(x_path)
            y_file = nib.load(y_path)
            x_data = x_file.get_fdata()
            y_data = y_file.get_fdata()
            # x_data = x_data / np.amax(x_data)

            # if train_dict["target_img"] == "MR":
            #     y_data = copy.deepcopy(x_data)

            # x_data = add_noise(
            #             x = x_data, 
            #             noise_type = train_dict["noise_type"],
            #             noise_params = train_dict["noise_params"],
            #             )

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
            
            if train_dict["n_MTGD"] == 1:
                
                if isTrain:

                    # optim.zero_grad()
                    # y_hat, y_cm = model(batch_x)
                    # cm_loss = loss_L2(y_cm, ONE_CM)
                    # recon_loss = loss_L1(batch_y, y_hat, y_cm)
                    # loss = cm_loss*train_dict["alpha_loss_CM"] + recon_loss*(1-train_dict["alpha_loss_CM"])
                    # loss.backward()
                    # optim.step()
                    # case_loss[cnt_file, 0] = recon_loss.item()
                    # case_loss[cnt_file, 1] = cm_loss.item()
                    # print("Loss: ", loss.item(), "Recon loss: ", recon_loss.item(), "CM loss", cm_loss.item())


                    optim.zero_grad()
                    y_hat = model_G(batch_x)
                    y_cm = model_E(torch.cat([batch_x, y_hat], axis=1))
                    loss_recon = nn.SmoothL1Loss()(batch_y, y_hat)
                    loss_weighted_recon = torch.mul(loss_recon, torch.sigmoid(y_cm))
                    loss_CM = nn.MSELoss()(y_cm, ONE_CM)
                    loss = loss_recon + loss_CM
                    loss.backward()
                    optim.step()
                    case_loss[cnt_file, 0] = torch.mean(loss_weighted_recon)
                    case_loss[cnt_file, 1] = loss_CM.item()
                    case_loss[cnt_file, 2] = loss_recon.item()
                    print("Loss: ", loss.item(),
                        "wRecon: ", case_loss[cnt_file, 0],
                        "CM: ", case_loss[cnt_file, 1],
                        "Recon: ", case_loss[cnt_file, 2])

            # if train_dict["n_MTGD"] > 1:

            #     if isTrain:

            #         average_loss = np.zeros((train_dict["n_MTGD"], 2))

            #         for idx_MTGD in range(train_dict["n_MTGD"]):
            #             optimizer.zero_grad()
            #             y_hat, y_CM = model(batch_x)
            #             L1 = criterion(batch_y, y_hat, y_CM)
            #             L2 = loss_CM(y_CM, ONE_CM)
            #             loss = L2*train_dict["alpha_loss_CM"] + L1*(1-train_dict["alpha_loss_CM"])
            #             loss.backward()
            #             average_loss[idx_MTGD, 0] = L1.item()
            #             average_loss[idx_MTGD, 1] = L2.item()

            #             for model_key, param in model.named_parameters():
            #                 # print(model_key, param.grad)
            #                 # print("-"*60)
            #                 MTGD_dict[model_key][idx_MTGD, :] = torch.flatten(param.grad).to("cpu").numpy()

            #         optimizer.zero_grad()
            #         for model_key, param in model.named_parameters():
            #             median_gradient = np.median(MTGD_dict[model_key], axis=0)
            #             median_gradient = np.reshape(median_gradient, param.grad.size())
            #             param.grad = torch.from_numpy(median_gradient).float().to(device)
            #         optimizer.step()

            #         case_loss[cnt_file, 0] = np.mean(average_loss[:, 0])
            #         case_loss[cnt_file, 1] = np.mean(average_loss[:, 1])
            #         print("Loss: ", loss.item(), "Recon loss: ", L1.item(), "CM loss", L2.item())

            if isVal:
                with torch.no_grad():

                    y_hat = model_G(batch_x)
                    y_cm = model_E(torch.cat([batch_x, y_hat], axis=1))
                    loss_recon = nn.SmoothL1Loss()(batch_y, y_hat)
                    loss_weighted_recon = torch.mul(loss_recon, torch.sigmoid(y_cm))
                    loss_CM = nn.MSELoss()(y_cm, ONE_CM)
                    loss = loss_recon + loss_CM
                case_loss[cnt_file, 0] = torch.mean(loss_weighted_recon)
                case_loss[cnt_file, 1] = loss_CM.item()
                case_loss[cnt_file, 2] = loss_recon.item()
                print("Loss: ", loss.item(),
                    "wRecon: ", case_loss[cnt_file, 0],
                    "CM: ", case_loss[cnt_file, 1],
                    "Recon: ", case_loss[cnt_file, 2])

        epoch_loss = np.mean(case_loss[cnt_file, 0])+np.mean(case_loss[cnt_file, 1])
        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("Loss: ", epoch_loss,
            "wRecon: ", np.mean(case_loss[cnt_file, 0]),
            "CM: ", np.mean(case_loss[cnt_file, 1]),
            "Recon: ", np.mean(case_loss[cnt_file, 2]))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_yhat.npy", y_hat.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_ycm.npy", y_cm.cpu().detach().numpy())
            torch.save(model, train_dict["save_folder"]+"model_.pth".format(idx_epoch + 1))
            
            if epoch_loss < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optim, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = epoch_loss

        # del batch_x, batch_y
        # gc.collect()
        # torch.cuda.empty_cache()
