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
test_dict["eval_step"] = [16, 16, 16] # <= input_size

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
X_list = data_div['test_list_X']

# ==================== Evaluating ====================

# wandb.watch(model)

file_list = X_list
iter_tag = "test"
cnt_total_file = len(file_list)
total_loss = 0
model.eval()

for cnt_file, file_path in enumerate(file_list):
    
    x_path = file_path
    y_path = file_path.replace("MR", "CT")
    file_name = os.path.basename(file_path)
    print(iter_tag + " ===> Case[{:03d}/{:03d}]: ".format(cnt_file+1, cnt_total_file), x_path, "<---", end="")
    x_file = nib.load(x_path)
    y_file = nib.load(y_path)
    x_data = x_file.get_fdata()
    y_data = y_file.get_fdata()
    ax, ay, az = x_data.shape
    case_loss = 0

    pad_x_data = np.pad(x_data, ((test_dict["input_size"][0] - test_dict["eval_step"][0],
                                  test_dict["input_size"][0] - test_dict["eval_step"][0]),
                                 (test_dict["input_size"][1] - test_dict["eval_step"][1],
                                  test_dict["input_size"][1] - test_dict["eval_step"][1]),
                                 (test_dict["input_size"][2] - test_dict["eval_step"][2],
                                  test_dict["input_size"][2] - test_dict["eval_step"][2])), 'constant')
    pad_y_data = np.pad(y_data, ((test_dict["input_size"][0] - test_dict["eval_step"][0],
                                  test_dict["input_size"][0] - test_dict["eval_step"][0]),
                                 (test_dict["input_size"][1] - test_dict["eval_step"][1],
                                  test_dict["input_size"][1] - test_dict["eval_step"][1]),
                                 (test_dict["input_size"][2] - test_dict["eval_step"][2],
                                  test_dict["input_size"][2] - test_dict["eval_step"][2])), 'constant')

    pad_y_hat = np.zeros(pad_y_data.shape)

    for ix in range((ax+test_dict["input_size"][0])//test_dict["eval_step"][0]-2):
        for iy in range((ay+test_dict["input_size"][1])//test_dict["eval_step"][1]-2):
            for iz in range((az+test_dict["input_size"][2])//test_dict["eval_step"][2]-2):
                sx = ix * test_dict["eval_step"][0]
                sy = iy * test_dict["eval_step"][1]
                sz = iz * test_dict["eval_step"][2]

                ex = sx + +test_dict["input_size"][0]
                ey = sy + +test_dict["input_size"][0]
                ez = sz + +test_dict["input_size"][0]

                batch_x = np.zeros((1, 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))
                batch_y = np.zeros((1, 1, train_dict["input_size"][0], train_dict["input_size"][1], train_dict["input_size"][2]))

                batch_x[0, 0, :] = pad_x_data[sx:ex, sy:ey, sz:ez]
                batch_x[0, 0, :] = pad_x_data[sx:ex, sy:ey, sz:ez]

                batch_x = torch.from_numpy(batch_x).float().to(device)
                batch_y = torch.from_numpy(batch_y).float().to(device)
        
                batch_z = model(batch_x)
                # print(batch_z.size(), batch_y.size())
                loss = loss_func(batch_z, batch_y)
                case_loss += loss.item
                
                pad_y_hat[sx:ex, sy:ey, sz:ez] += batch_z.cpu().detach().numpy()
                del batch_x, batch_y
                gc.collect()
                torch.cuda.empty_cache()

    case_loss /= (ix*iy*iz) # maximum ix/iy/iz after iteration
    total_loss += case_loss
    print(" ->", train_dict['loss_term'], case_loss)

    cnt_each_cube = 1
    cnt_each_cube *= test_dict["input_size"][0]//test_dict["eval_step"][0]
    cnt_each_cube *= test_dict["input_size"][1]//test_dict["eval_step"][1]
    cnt_each_cube *= test_dict["input_size"][2]//test_dict["eval_step"][2]
    pad_y_hat /= cnt_each_cube

    pad_y_hat = pad_y_hat[test_dict["input_size"][0]-test_dict["eval_step"][0]:test_dict["eval_step"][0]-test_dict["input_size"][0],
                          test_dict["input_size"][1]-test_dict["eval_step"][1]:test_dict["eval_step"][1]-test_dict["input_size"][1],
                          test_dict["input_size"][2]-test_dict["eval_step"][2]:test_dict["eval_step"][2]-test_dict["input_size"][2],
                          ]

    test_file = nib.Nifti1Image(pad_y_hat, x_file.affine, x_file.header)
    test_save_name = train_dict["save_folder"]+"pred/"+file_name
    nib.save(test_file, test_save_name)

total_loss /= cnt_total_file
print("Total ", train_dict['loss_term'], total_loss)
np.save(train_dict["save_folder"]+"pred/", os.path.basename(model_list[-1])+"_total_loss.npy", total_loss)



