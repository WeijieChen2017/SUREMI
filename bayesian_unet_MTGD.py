baimport os
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

from monai.networks.nets.unet import UNet as UNet
from monai.networks.layers.factories import Act, Norm
import bnn

class UnetBNN(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, unet_dict):
        super().__init__()
    
        self.unet = UNet( 
            spatial_dims=unet_dict["spatial_dims"],
            in_channels=unet_dict["in_channels"],
            out_channels=unet_dict["mid_channels"],
            channels=unet_dict["channels"],
            strides=unet_dict["strides"],
            num_res_units=unet_dict["num_res_units"],
            act=unet_dict["act"],
            norm=unet_dict["normunet"],
            dropout=unet_dict["dropout"],
            bias=unet_dict["bias"],
            )
        if unet_dict["spatial_dims"] == 2:
            self.out_conv = nn.Conv2d(
                unet_dict["mid_channels"],
                unet_dict["out_channels"], 
                kernel_size=1
                )
        if unet_dict["spatial_dims"] == 3:
            self.out_conv = nn.Conv3d(
                unet_dict["mid_channels"],
                unet_dict["out_channels"], 
                kernel_size=1
                )

        bnn.bayesianize_(
            self.out_conv,
            inference=unet_dict["inference"],
            inducing_rows=unet_dict["inducing_rows"],
            inducing_cols=unet_dict["inducing_cols"],
            )

    def forward(self, x):
        x = self.unet(x)
        x = self.out_conv(x)
        return x

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "Bayesian_unet_v17_unet_BNN_KLe6_MTGD5"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["input_size"] = [96, 96, 96]
train_dict["gpu_ids"] = [1]
train_dict["epochs"] = 200
train_dict["batch"] = 8
train_dict["dropout"] = 0
train_dict["beta"] = 1e6 # resize KL loss
train_dict["model_term"] = "Monai_Unet3d"
train_dict["dataset_ratio"] = 0.25
train_dict["continue_training_epoch"] = 0
train_dict["flip"] = False
train_dict["n_MTGD"] = 5

unet_dict = {}
unet_dict["spatial_dims"] = 3
unet_dict["in_channels"] = 1
unet_dict["mid_channels"] = 64
unet_dict["out_channels"] = 1
unet_dict["channels"] = (32, 64, 128, 256)
unet_dict["strides"] = (2, 2, 2)
unet_dict["num_res_units"] = 4
unet_dict["inference"] = "inducing"
unet_dict["inducing_rows"] = 64
unet_dict["inducing_cols"] = 64
unet_dict["act"] = Act.PRELU
unet_dict["normunet"] = Norm.INSTANCE
unet_dict["dropout"] = 0.0
unet_dict["bias"] = True

train_dict["model_related"] = unet_dict

train_dict["folder_X"] = "./data_dir/Iman_MR/norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/norm/"
# train_dict["pre_train"] = "swin_base_patch244_window1677_kinetics400_22k.pth"
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

# model = UNet( 
#     spatial_dims=unet_dict["spatial_dims"],
#     in_channels=unet_dict["in_channels"],
#     out_channels=unet_dict["out_channels"],
#     channels=unet_dict["channels"],
#     strides=unet_dict["strides"],
#     num_res_units=unet_dict["num_res_units"],
#     act=unet_dict["act"],
#     norm=unet_dict["normunet"],
#     dropout=unet_dict["dropout"],
#     bias=unet_dict["bias"],
#     )

# bnn.bayesianize_(model, inference="inducing", inducing_rows=64, inducing_cols=64)

# model = torch.load(train_dict["save_folder"]+"model_best_{:03d}".format(
#     train_dict["continue_training_epoch"])+".pth", map_location=torch.device('cpu'))
# bnn.bayesianize_(model, inference="inducing", inducing_rows=64, inducing_cols=64)
# optimizer = torch.load(train_dict["save_folder"]+"optim_{:03d}".format(
#     train_dict["continue_training_epoch"])+".pth")
model = UnetBNN(unet_dict)
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

MTGD_dict = {}
    for model_key, param in model.named_parameters():
        new_shape = (n_MTGD, torch.flatten(param).size()[0])
        MTGD_dict[model_key] = np.zeros(new_shape)

# ==================== training ====================

best_val_loss = 1e3
best_epoch = 0
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
            model.train()
        else:
            model.eval()

        random.shuffle(file_list)
        
        case_loss = np.zeros((len(file_list), 2))

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
            x_data = x_data / np.amax(x_data)

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
                for idx_MTGD in range(train_dict["n_MTGD"]):
                    optimizer.zero_grad()
                    y_hat = model(batch_x)
                    L1 = criterion(y_hat, batch_y)
                    kl = sum(m.kl_divergence() for m in model.out_conv.modules() if hasattr(m, "kl_divergence"))
                    kl /= len(file_list)
                    if not train_dict["flip"]:
                        loss = L1 + kl / train_dict["beta"]
                    else:
                        if idx_epoch % 2 == 0:
                            loss = L1
                        else:
                            loss = kl / train_dict["beta"]
                    # loss = L1
                    loss.backward()

                    for model_key, param in model.named_parameters():
                        MTGD_dict[model_key][idx_MTGD, :] = torch.flatten(param.grad).numpy().to("cpu")

                optimizer.zero_grad()
                for model_key, param in model.named_parameters():
                    median_gradient = np.median(MTGD_dict[model_key], axis=0)
                    median_gradient = np.reshape(median_gradient, param.grad.size())
                    param.grad = torch.from_numpy(median_gradient).float().to(device)

                optimizer.step()
                case_loss[cnt_file, 0] = L1.item()
                case_loss[cnt_file, 1] = kl.item()
                print("Loss: ", loss.item(), "KL: ", kl.item(), "L1:", L1.item())

            if isVal:
                with torch.no_grad():
                    y_hat = model(batch_x)
                    L1 = criterion(y_hat, batch_y)
                    kl = sum(m.kl_divergence() for m in model.out_conv.modules() if hasattr(m, "kl_divergence"))
                    kl /= len(file_list)
                    if not train_dict["flip"]:
                        loss = L1 + kl / train_dict["beta"]
                    else:
                        if idx_epoch % 2 == 0:
                            L1loss = L1
                        else:
                            loss = kl / train_dict["beta"]
                case_loss[cnt_file, 0] = L1.item()
                case_loss[cnt_file, 1] = kl.item()
                print("Loss: ", loss.item(), "KL: ", kl.item(), "L1:", L1.item())

        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss), "  Recon: ", np.mean(case_loss[:, 0]))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

            torch.save(model, train_dict["save_folder"]+"model_.pth".format(idx_epoch + 1))
            if np.mean(case_loss[:, 0]) < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optimizer, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = np.mean(case_loss[:, 0])

        del batch_x, batch_y
        gc.collect()
        torch.cuda.empty_cache()
