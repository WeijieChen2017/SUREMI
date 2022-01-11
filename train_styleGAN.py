import os
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

from model.styleGAN_MR2CT import Generator

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "z_to_img"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_channel"] = 3
train_dict["output_channel"] = 3
train_dict["gpu_ids"] = [7]
train_dict["epochs"] = 50
train_dict["batch"] = 10
train_dict["dropout"] = 0
train_dict["model_term"] = "styleGAN"

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


train_dict["latent_size"] = 512
train_dict["dlatent_size"] = 512
train_dict["resolution"] = 256
train_dict["structure"] = "fixed"
train_dict["style_mixing_prob"] = 0.0
train_dict["depth"] = int(np.log2(train_dict["resolution"])) - 1



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

gen_MR = Generator(
    latent_size=train_dict["latent_size"],
    dlatent_size=train_dict["dlatent_size"],
    num_channels=train_dict["input_channel"],
    resolution=train_dict["resolution"],
    structure=train_dict["structure"],
    style_mixing_prob=train_dict["style_mixing_prob"]).train().to(device)

gen_CT = Generator(
    latent_size=train_dict["latent_size"],
    dlatent_size=train_dict["dlatent_size"],
    num_channels=train_dict["input_channel"],
    resolution=train_dict["resolution"],
    structure=train_dict["structure"],
    style_mixing_prob=train_dict["style_mixing_prob"]).train().to(device)

criterion_MR = nn.SmoothL1Loss()
criterion_CT = nn.SmoothL1Loss()

opt_MR = torch.optim.AdamW(
    gen_MR.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

opt_CT = torch.optim.AdamW(
    gen_CT.parameters(),
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

best_val_loss_MR = 1e6
best_val_loss_CT = 1e6
rand_dict = {}
# wandb.watch(model)

for idx_epoch in range(train_dict["epochs"]):
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))
    np.save(train_dict["save_folder"]+"rand_dict.npy", rand_dict)           

    package_train = [train_list_X, True, False, "train"]
    package_val = [val_list_X, False, True, "val"]
    package_test = [test_list_X, False, False, "test"]

    for package in [package_train, package_val, package_test]:

        file_list = package[0]
        isTrain = package[1]
        isVal = package[2]
        iter_tag = package[3]

        if isTrain:
            gen_MR.train()
            gen_CT.train()
        else:
            gen_MR.eval()
            gen_CT.eval()

        random.shuffle(file_list)
        epoch_loss_MR = np.zeros((len(file_list)))
        epoch_loss_CT = np.zeros((len(file_list)))
        for cnt_file, file_path in enumerate(file_list):
            
            file_name = os.path.basename(file_path)
            cube_x_path = file_path
            cube_y_path = train_dict["folder_Y"] + file_name
            print("--->",cube_x_path,"<---", end="")
            cube_x_data = nib.load(cube_x_path).get_fdata()
            cube_y_data = nib.load(cube_y_path).get_fdata()
            len_z = cube_x_data.shape[2]
            case_loss_MR = np.zeros((len_z//train_dict["batch"]))
            case_loss_CT = np.zeros((len_z//train_dict["batch"]))
            input_list = list(range(len_z))
            random.shuffle(input_list)

            for idx_iter in range(len_z//train_dict["batch"]):

                batch_x = np.zeros((train_dict["batch"], train_dict["input_channel"], cube_x_data.shape[0], cube_x_data.shape[1]))
                batch_y = np.zeros((train_dict["batch"], train_dict["output_channel"], cube_y_data.shape[0], cube_y_data.shape[1]))
                batch_seed = np.zeros((train_dict["batch"], train_dict["latent_size"]))

                for idx_batch in range(train_dict["batch"]):
                    z_center = input_list[idx_iter*train_dict["batch"]+idx_batch]
                    z_before = z_center - 1 if z_center > 0 else 0
                    z_after = z_center + 1 if z_center < len_z-1 else len_z-1

                    if train_dict["input_channel"] == 3:
                        batch_x[idx_batch, 0, :, :] = cube_x_data[:, :, z_before]
                        batch_x[idx_batch, 1, :, :] = cube_x_data[:, :, z_center]
                        batch_x[idx_batch, 2, :, :] = cube_x_data[:, :, z_after]
                        
                    if train_dict["output_channel"] == 3:
                        batch_y[idx_batch, 0, :, :] = cube_y_data[:, :, z_center]
                        batch_y[idx_batch, 1, :, :] = cube_y_data[:, :, z_center]
                        batch_y[idx_batch, 2, :, :] = cube_y_data[:, :, z_center]

                    rand_key = file_name + "_" + str(z_center)
                    if not rand_key in rand_dict:
                        rand_dict[rand_key] = torch.randn(1, train_dict["latent_size"])
                    batch_seed[idx_batch, :] = rand_dict[rand_key]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
                batch_seed = torch.from_numpy(batch_seed).float().to(device)

                # ------------------------------start to train------------------------------

                opt_MR.zero_grad()
                opt_CT.zero_grad()
                MR_hat = gen_MR(batch_seed, train_dict["depth"], 1)
                CT_hat = gen_CT(batch_seed, train_dict["depth"], 1)
                loss_MR = criterion_MR(MR_hat, batch_x)
                loss_CT = criterion_MR(CT_hat, batch_y)
                if isTrain:
                    loss_MR.backward()
                    loss_CT.backward()
                    opt_MR.step()
                    opt_CT.step()


                # ------------------------------end of train------------------------------

                case_loss_MR[idx_iter] = loss.item()
                case_loss_CT[idx_iter] = loss.item()
            
            case_name = os.path.basename(cube_x_path)[5:8]
            if not isTrain:
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_x.npy".format(idx_epoch+1, case_name), batch_x.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_y.npy".format(idx_epoch+1, case_name), batch_y.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_seed.npy".format(idx_epoch+1, case_name), batch_seed.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_x_hat.npy".format(idx_epoch+1, case_name), MR_hat.cpu().detach().numpy())
                np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_"+iter_tag+"_y_hat.npy".format(idx_epoch+1, case_name), CT_hat.cpu().detach().numpy())

            # after training one case
            loss_mean_MR = np.mean(case_loss_MR)
            loss_std_MR = np.std(case_loss_MR)
            loss_mean_CT = np.mean(case_loss_CT)
            loss_std_CT = np.std(case_loss_CT)
            print("===> Epoch[{:03d}]-Case[{:03d}]: ".format(idx_epoch+1, cnt_file+1), end='')
            print("Loss MR mean: {:.6} Loss std: {:.6}".format(loss_mean_MR, loss_std_MR))
            print("Loss CT mean: {:.6} Loss std: {:.6}".format(loss_mean_CT, loss_std_CT))
            epoch_loss_MR[cnt_file] = loss_mean_MR
            epoch_loss_CT[cnt_file] = loss_mean_CT
        
            
        # after training all cases
        loss_mean_MR = np.mean(epoch_loss_MR)
        loss_std_MR = np.std(epoch_loss_MR)
        loss_mean_CT = np.mean(epoch_loss_CT)
        loss_std_CT = np.std(epoch_loss_CT)
        
        print("===> Epoch[{}]: ".format(idx_epoch+1), end='')
        print("Loss MR mean: {:.6} Loss std: {:.6}".format(loss_mean_MR, loss_std_MR))
        print("Loss CT mean: {:.6} Loss std: {:.6}".format(loss_mean_CT, loss_std_CT))
        # wandb.log({iter_tag+"_loss_MR": loss_mean_MR})
        # wandb.log({iter_tag+"_loss_CT": loss_mean_CT})
        np.save(train_dict["save_folder"]+"loss/epoch_loss_MR_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), epoch_loss_MR)
        np.save(train_dict["save_folder"]+"loss/epoch_loss_CT_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), epoch_loss_CT)

        if isVal:
            if loss_mean_MR < best_val_loss_MR:
                # save the best model
                torch.save(gen_MR, train_dict["save_folder"]+"model_best_MR_{:03d}.pth".format(idx_epoch+1))
                print("Checkpoint MR saved at Epoch {:03d}".format(idx_epoch+1))
                best_val_loss_MR = loss_mean_MR

            if loss_mean_CT < best_val_loss_CT:
                # save the best model
                torch.save(gen_CT, train_dict["save_folder"]+"model_best_CT_{:03d}.pth".format(idx_epoch+1))
                print("Checkpoint CT saved at Epoch {:03d}".format(idx_epoch+1))
                best_val_loss_CT = loss_mean_CT

        torch.cuda.empty_cache()
