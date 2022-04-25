import os
import gc
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

# from model import TransformerModel
from torch.nn import Transformer

# ==================== dict and config ====================

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "VQ_8x_v1"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
# train_dict["input_channel"] = 30
# train_dict["output_channel"] = 30
train_dict["gpu_ids"] = [7]
train_dict["epochs"] = 100
train_dict["batch"] = 32
train_dict["dropout"] = 0
train_dict["model_term"] = "TransformerModel"

train_dict["folder_X"] = "./data_dir/Iman_MR/VQ3d/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/VQ3d/"
train_dict["val_ratio"] = 0.375

train_dict["loss_term"] = "CrossEntropyLoss"
train_dict["optimizer"] = "AdamW"
train_dict["opt_lr"] = 1e-3 # default
train_dict["opt_betas"] = (0.9, 0.999) # default
train_dict["opt_eps"] = 1e-8 # default
train_dict["opt_weight_decay"] = 0.01 # default
train_dict["amsgrad"] = False # default

train_dict["model_related"] = {}
train_dict["model_related"]["d_model"] = 4096, 
train_dict["model_related"]["nhead"] = 8, 
train_dict["model_related"]["num_encoder_layers"]=4, 
train_dict["model_related"]["num_decoder_layers"]=4, 
train_dict["model_related"]["dim_feedforward"]=1024, 
train_dict["model_related"]["dropout"]=0.1, 
train_dict["model_related"]["activation"]=F.relu,

for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(train_dict["save_folder"]+"dict.npy", train_dict)


# ==================== basic settings ====================

np.random.seed(train_dict["seed"])
gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformer
model = Transformer(
    d_model=train_dict["model_related"]["d_model"][0], 
    nhead=train_dict["model_related"]["nhead"][0], 
    num_encoder_layers=train_dict["model_related"]["num_encoder_layers"][0], 
    num_decoder_layers=train_dict["model_related"]["num_decoder_layers"][0], 
    dim_feedforward=train_dict["model_related"]["dim_feedforward"][0], 
    dropout=train_dict["model_related"]["dropout"][0], 
    activation=train_dict["model_related"]["activation"][0], 
    custom_encoder=None, 
    custom_decoder=None, 
    layer_norm_eps=1e-05, 
    batch_first=False, 
    norm_first=False, 
    device=None, 
    dtype=None
    )

model.train()
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = train_dict["opt_lr"],
    betas = train_dict["opt_betas"],
    eps = train_dict["opt_eps"],
    weight_decay = train_dict["opt_weight_decay"],
    amsgrad = train_dict["amsgrad"]
    )

# ==================== data division ====================

DATA_x = np.load(train_dict["folder_X"]+"onehot_x_cube_8x.npy", allow_pickle=True)
DATA_y = np.load(train_dict["folder_Y"]+"onehot_y_cube_8x.npy", allow_pickle=True)
print("Load data as ", DATA_x.shape)
n_file = DATA_x.shape[0]
file_list = list(range(n_file))
random.shuffle(file_list)

FILENAME_X = list(DATA_x[:, 0])
FILENAME_Y = list(DATA_y[:, 0])
CODE_x = DATA_x[:, 1]
CODE_y = DATA_y[:, 1]
MAX_LEN_CASE = max(len(item) for item in CODE_x)

val_list = file_list[:int(n_file*train_dict["val_ratio"])]
val_list.sort()
train_list = list(set(file_list)-set(val_list))
train_list.sort()

data_division_dict = {
    "train_list" : train_list,
    "val_list" : val_list,
    "filename_list_X" : FILENAME_X,
    "filename_list_Y" : FILENAME_Y}
np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)


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
best_epoch = 0

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
        cnt_total_file = len(file_list)
        case_loss = np.zeros((cnt_total_file))
        
        for cnt_file, idx_file in enumerate(file_list):
            
            x_path = FILENAME_X[idx_file]
            y_path = FILENAME_Y[idx_file]
            file_name = os.path.basename(file_path)
            print(iter_tag + " ===> Epoch[{:03d}]->[{:03d}]/[{:03d}]: --->".format(
                idx_epoch+1, 
                cnt_file+1,
                cnt_total_file), x_path, "<---", end="")
            x_data = DATA_x[idx_file]
            y_data = DATA_y[idx_file]

            # len, num_feature
            batch_x = np.zeros((len(x_data), 4096))
            batch_y = np.zeros((len(y_data), 4096))
            
            # # [CLS] [SEP]
            # batch_x[0, 0], batch_x[-1, -1] = 1, 1
            # batch_y[0, 0], batch_y[-1, -1] = 1, 1

            for idx_onehot in range(len(x_data)):
                batch_x[idx_onehot, x_data[idx_onehot]+1] = 1
                batch_y[idx_onehot, y_data[idx_onehot]+1] = 1

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
                optimizer.step()
            case_loss[cnt_file] = loss.item()
            print("Loss: ", case_loss[cnt_file])

        print(iter_tag + " ===>===> Epoch[{:03d}]: ".format(idx_epoch+1), end='')
        print("  Loss: ", np.mean(case_loss))
        np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}.npy".format(idx_epoch+1), case_loss)

        if isVal:
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_x.npy", batch_x.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_y.npy", batch_y.cpu().detach().numpy())
            np.save(train_dict["save_folder"]+"npy/Epoch[{:03d}]_Case[{}]_".format(idx_epoch+1, file_name)+iter_tag+"_z.npy", y_hat.cpu().detach().numpy())

            # torch.save(model, train_dict["save_folder"]+"model_.pth".format(idx_epoch + 1))
            if np.mean(case_loss) < best_val_loss:
                # save the best model
                torch.save(model, train_dict["save_folder"]+"model_best_{:03d}.pth".format(idx_epoch + 1))
                torch.save(optimizer, train_dict["save_folder"]+"optim_{:03d}.pth".format(idx_epoch + 1))
                print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                best_val_loss = np.mean(case_loss)

        del batch_x, batch_y
        gc.collect()
        torch.cuda.empty_cache()
