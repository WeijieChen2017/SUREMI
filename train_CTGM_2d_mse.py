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

from model import ComplexTransformerGenerationModel as CTGM

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "CTGM_2d_v2_mse"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
ax, ay = train_dict["input_size"]
train_dict["gpu_ids"] = [7]
train_dict["epochs"] = 500
train_dict["batch"] = 16
train_dict["dropout"] = 0
train_dict["model_term"] = "ComplexTransformerGenerationModel"

train_dict["model_related"] = {}
train_dict["model_related"]["cx"] = 32
cx = train_dict["model_related"]["cx"]
train_dict["model_related"]["input_dims"] = [cx**2, cx**2]
train_dict["model_related"]["hidden_size"] = 1024
train_dict["model_related"]["embed_dim"] = 1024
train_dict["model_related"]["output_dim"] = cx**2*2
train_dict["model_related"]["num_heads"] = cx
train_dict["model_related"]["attn_dropout"] = 0.0
train_dict["model_related"]["relu_dropout"] = 0.0
train_dict["model_related"]["res_dropout"] = 0.0
train_dict["model_related"]["out_dropout"] = 0.0
train_dict["model_related"]["layers"] = 6
train_dict["model_related"]["attn_mask"] = False

train_dict["folder_X"] = "./data_dir/Iman_MR/kspace_2d/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/kspace_2d/"
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

# model = CTGM( 
#     input_dims=train_dict["model_related"]["input_dims"],
#     hidden_size=train_dict["model_related"]["hidden_size"],
#     embed_dim=train_dict["model_related"]["embed_dim"],
#     output_dim=train_dict["model_related"]["output_dim"],
#     num_heads=train_dict["model_related"]["num_heads"],
#     attn_dropout=train_dict["model_related"]["attn_dropout"],
#     relu_dropout=train_dict["model_related"]["relu_dropout"],
#     res_dropout=train_dict["model_related"]["res_dropout"],
#     out_dropout=train_dict["model_related"]["out_dropout"],
#     layers=train_dict["model_related"]["layers"],
#     attn_mask=train_dict["model_related"]["attn_mask"])

model = torch.load(train_dict["save_folder"]+"model_best_102.pth")

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

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.npy"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.npy"))

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

best_val_loss = 1000
best_epoch = 0
# wandb.watch(model)

# package_train = [train_list[:1], True, False, "train"]
package_train = [train_list, True, True, "train"]
package_val = [val_list, False, True, "val"]
# package_test = [test_list, False, False, "test"]

num_vocab = (ax//cx) * (ay//cx)

for idx_epoch_new in range(train_dict["epochs"]):
    idx_epoch = idx_epoch_new + 108
    print("~~~~~~Epoch[{:03d}]~~~~~~".format(idx_epoch+1))

    for package in [package_train, package_val]: # , package_val

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

        """
        x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
        """

        for cnt_file, file_path in enumerate(file_list):
            
            total_file = len(file_list)
            
            x_path = file_path
            y_path = file_path.replace("MR", "CT")
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]-[{:03d}]/[{:03d}]: --->".format(idx_epoch+1, cnt_file+1, total_file), file_name, "<---", end="") #
            x_data = np.load(x_path)
            y_data = np.load(y_path)
            dz = x_data.shape[0]
            z_list = list(range(dz))
            random.shuffle(z_list)
            batch_per_step = train_dict["batch"]
            # batch_per_step = dz
            batch_loss = np.zeros((dz // batch_per_step))

            for ib in range(dz // batch_per_step):

                batch_x = np.zeros((num_vocab, batch_per_step, cx**2*2))
                batch_y = np.zeros((num_vocab, batch_per_step, cx**2*2))
                batch_offset = ib * batch_per_step

                for iz in range(batch_per_step):

                    batch_x[:, iz, :] = x_data[z_list[iz+batch_offset], :, :]
                    batch_y[:, iz, :] = y_data[z_list[iz+batch_offset], :, :]

                batch_x = torch.from_numpy(batch_x).float().to(device).contiguous()
                batch_y = torch.from_numpy(batch_y).float().to(device).contiguous()
                    
                optimizer.zero_grad()
                # print(batch_x.size(), batch_y.size())
                y_hat = model(batch_x, batch_y)
                # print("Yhat size: ", y_hat.size(), end="   ")
                # print("Ytrue size: ", batch_y.size())
                loss = criterion(y_hat, batch_y)
                if isTrain:
                    loss.backward()
                    optimizer.step()
                batch_loss[ib] = loss.item()

            case_loss[cnt_file] = np.mean(batch_loss)
            print("Loss: ", case_loss[cnt_file])

            if cnt_file < len(file_list)-1:
                del batch_x, batch_y
                gc.collect()
                torch.cuda.empty_cache()

        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)



        # if idx_epoch % 10 == 1:
        #     np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
        #     np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
        #     np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())
        #     torch.save(model, train_dict["save_folder"]+"model_{:03d}.pth".format(idx_epoch + 1))
        if isVal:
            if np.mean(case_loss) < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = np.mean(case_loss)

        del batch_x, batch_y
        gc.collect()
        torch.cuda.empty_cache()
