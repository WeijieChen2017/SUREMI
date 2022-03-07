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

from model import SwinTransformer3D

# ==================== dict and config ====================

test_dict = {}
test_dict = {}
test_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
test_dict["project_name"] = "Unet_Monai_Iman"
test_dict["save_folder"] = "./project_dir/"+test_dict["project_name"]+"/"
test_dict["gpu_ids"] = [7]
test_dict["eval_step"] = [32, 32, 32] # <= input_size
test_dict["eval_file_cnt"] = 16
test_dict["fusion_method"] = "median" # sum or median

train_dict = np.load(test_dict["save_folder"]+"dict.npy", allow_pickle=True)[()]

test_dict["seed"] = train_dict["seed"]
test_dict["input_size"] = train_dict["input_size"]


for path in [test_dict["save_folder"], test_dict["save_folder"]+"pred/"]:
    if not os.path.exists(path):
        os.mkdir(path)

np.save(test_dict["save_folder"]+"test_dict.npy", test_dict)


# ==================== basic settings ====================

np.random.seed(test_dict["seed"])
gpu_list = ','.join(str(x) for x in test_dict["gpu_ids"])
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_list = sorted(glob.glob(os.path.join(test_dict["save_folder"], "model_best_*.pth")))
model = torch.load(model_list[-1], map_location=torch.device('cpu'))
print("--->", model_list[-1], " is loaded.")

model = model.to(device)
# loss_func = getattr(nn, train_dict['loss_term'])
loss_func = nn.SmoothL1Loss()

# ==================== data division ====================

data_div = np.load(os.path.join(test_dict["save_folder"], "data_division.npy"), allow_pickle=True)[()]
X_list = data_div['test_list_X'][6:test_dict["eval_file_cnt"]]

# ==================== Evaluating ====================

# wandb.watch(model)

file_list = X_list
iter_tag = "test"
cnt_total_file = len(file_list)
total_loss = 0
step_x, step_y, step_z = test_dict["eval_step"]
ins_x, ins_y, ins_z = test_dict["input_size"]
model.eval()

cnt_each_cube = 1
cnt_each_cube *= ins_x//step_x
cnt_each_cube *= ins_y//step_y
cnt_each_cube *= ins_z//step_z

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="")
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    x_data = x_data / np.amax(x_data)
    y_data = y_file.get_fdata()
    ax, ay, az = x_data.shape
    case_loss = 0

    pad_x_data = np.pad(x_data, ((ins_x - step_x, ins_x - step_x),
                                 (ins_y - step_y, ins_y - step_y),
                                 (ins_z - step_z, ins_z - step_z)), 'constant')
    pad_y_data = np.pad(y_data, ((ins_x - step_x, ins_x - step_x),
                                 (ins_y - step_y, ins_y - step_y),
                                 (ins_z - step_z, ins_z - step_z)), 'constant')

    pad_y_hat = np.zeros((cnt_each_cube, pad_y_data.shape[0], pad_y_data.shape[1], pad_y_data.shape[2]))
    step_x_cnt = (ax+ins_x)//step_x-2
    step_y_cnt = (ay+ins_y)//step_y-2
    step_z_cnt = (az+ins_z)//step_z-2
    cnt_cube_y_hat = np.zeros(((ax+ins_x)//step_x, (ay+ins_y)//step_y, (az+ins_z)//step_z), dtype=np.int32)

    for ix in range(step_x_cnt):
        for iy in range(step_y_cnt):
            for iz in range(step_z_cnt):
                sx = ix * step_x
                sy = iy * step_y
                sz = iz * step_z

                ex = sx + +ins_x
                ey = sy + +ins_x
                ez = sz + +ins_x

                batch_x = np.zeros((1, 1, ins_x, ins_y, ins_z))
                batch_y = np.zeros((1, 1, ins_x, ins_y, ins_z))

                batch_x[0, 0, :] = pad_x_data[sx:ex, sy:ey, sz:ez]
                batch_x[0, 0, :] = pad_x_data[sx:ex, sy:ey, sz:ez]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
        
                batch_z = model(batch_x)
                loss = loss_func(batch_z, batch_y)
                case_loss += loss.item()
                
                # pad_y_hat[sx:ex, sy:ey, sz:ez] += np.squeeze(batch_z.cpu().detach().numpy())
                batch_z = np.squeeze(batch_z.cpu().detach().numpy())
                for iix in range(ins_x//step_x):
                    for iiy in range(ins_y//step_y):
                        for iiz in range(ins_z//step_z):
                            # print()
                            # print(step_x_cnt, step_y_cnt, step_z_cnt)
                            # print(ins_x//step_x, ins_y//step_z, ins_y//step_z)
                            curr_idx = cnt_cube_y_hat[ix+iix, iy+iiy, iz+iiz]
                            bz_x = step_x * iix
                            bz_y = step_y * iiy
                            bz_z = step_z * iiz
                            cube_batch_z = batch_z[bz_x:bz_x+step_x,
                                                   bz_y:bz_y+step_y,
                                                   bz_z:bz_z+step_z]
                            pyh_x = sx+bz_x
                            pyh_y = sy+bz_y
                            pyh_z = sz+bz_z
                            pad_y_hat[curr_idx, pyh_x:pyh_x+step_x,
                                                pyh_y:pyh_y+step_y,
                                                pyh_z:pyh_z+step_z] = cube_batch_z

                            cnt_cube_y_hat[ix+iix, iy+iiy, iz+iiz] += 1

                del batch_x, batch_y
                gc.collect()
                torch.cuda.empty_cache()

    case_loss /= (ix*iy*iz) # maximum ix/iy/iz after iteration
    total_loss += case_loss
    print(" ->", train_dict['loss_term'], case_loss)

    if test_dict["fusion_method"] == "median":
        pad_y_hat = np.squeeze(np.median(pad_y_hat), axis=0)
    if test_dict["fusion_method"] == "mean":
        pad_y_hat = np.squeeze(np.mean(pad_y_hat), axis=0)    

    pad_y_hat = pad_y_hat[int(ins_x-step_x):int(step_x-ins_x),
                          int(ins_y-step_y):int(step_y-ins_y),
                          int(ins_z-step_z):int(step_z-ins_z)]

    test_file = nib.Nifti1Image(pad_y_hat, x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+"pred/"+file_name
    nib.save(test_file, test_save_name)

total_loss /= cnt_total_file
print("Total ", train_dict['loss_term'], total_loss)
np.save(train_dict["save_folder"]+"pred/", os.path.basename(model_list[-1])+"_total_loss.npy", total_loss)



