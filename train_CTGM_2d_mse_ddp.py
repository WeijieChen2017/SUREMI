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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import ComplexTransformerGenerationModel as CTGM

train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "CTGM_2d_v8_mse_maxlen_ddp"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
train_dict["seed"] = 426
train_dict["input_size"] = [256, 256]
ax, ay = train_dict["input_size"]
train_dict["gpu_ids"] = [1,2,5,6]
train_dict["epochs"] = 600
train_dict["batch"] = 1
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

train_dict["folder_X"] = "./data_dir/Iman_MR/kspace_2d_norm/"
train_dict["folder_Y"] = "./data_dir/Iman_CT/kspace_2d_norm/"
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


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    best_val_loss = 3

    # create model and move it to GPU with id rank

    # model = torch.load("./project_dir/CTGM_2d_v2_mse/model_best_102.pth").to(rank)

    model = CTGM( 
        input_dims=train_dict["model_related"]["input_dims"],
        hidden_size=train_dict["model_related"]["hidden_size"],
        embed_dim=train_dict["model_related"]["embed_dim"],
        output_dim=train_dict["model_related"]["output_dim"],
        num_heads=train_dict["model_related"]["num_heads"],
        attn_dropout=train_dict["model_related"]["attn_dropout"],
        relu_dropout=train_dict["model_related"]["relu_dropout"],
        res_dropout=train_dict["model_related"]["res_dropout"],
        out_dropout=train_dict["model_related"]["out_dropout"],
        layers=train_dict["model_related"]["layers"],
        attn_mask=train_dict["model_related"]["attn_mask"]
    ).to(rank)

    ddp_model = DDP(model, device_ids=[rank]) # , find_unused_parameters=True
    print("The model has been set at", rank)
    model.train()
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr = train_dict["opt_lr"],
        betas = train_dict["opt_betas"],
        eps = train_dict["opt_eps"],
        weight_decay = train_dict["opt_weight_decay"],
        amsgrad = train_dict["amsgrad"]
        )


    # package_train = [train_list[:1], True, False, "train"]
    package_train = [train_list, True, False, "train"]
    package_val = [val_list, False, True, "val"]
    # package_test = [test_list, False, False, "test"]

    num_vocab = (ax//cx) * (ay//cx)

    for idx_epoch_new in range(train_dict["epochs"]):
        idx_epoch = idx_epoch_new + 0

        # print("~Epoch[{:03d}]~".format(idx_epoch+1))
        
        for package in [package_train]: # , package_val

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
            total_file = len(file_list)
            total_group = len(file_list)//world_size
            case_loss = np.zeros(len(file_list)//world_size * world_size)
            """
            x should have dimension [seq_len, batch_size, n_features] (i.e., L, N, C).
            """

            for idx_file_group in range(len(file_list)//world_size):

                # print("----"*rank*3, "Line 179 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank)
                file_path = file_list[idx_file_group * 4 + rank]
                x_path = file_path
                y_path = file_path.replace("MR", "CT")
                file_name = os.path.basename(file_path)
                x_data = np.load(x_path)
                y_data = np.load(y_path)
                dz = x_data.shape[0]
                z_list = list(range(dz))
                random.shuffle(z_list)
                batch_per_step = train_dict["batch"]
                # batch_per_step = dz
                batch_loss = np.zeros((dz // batch_per_step, world_size))
                # print("----"*rank*3, "Line 192 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank)
                for ib in range(dz // batch_per_step):

                    batch_x = np.zeros((num_vocab, batch_per_step, cx**2*2))
                    batch_y = np.zeros((num_vocab, batch_per_step, cx**2*2))
                    batch_offset = ib * batch_per_step

                    for iz in range(batch_per_step):

                        batch_x[:, iz, :] = x_data[z_list[iz+batch_offset], :, :]
                        batch_y[:, iz, :] = y_data[z_list[iz+batch_offset], :, :]

                    batch_x = torch.from_numpy(batch_x).float().to(rank) # .contiguous()
                    batch_y = torch.from_numpy(batch_y).float().to(rank) # .contiguous()
                    # print("----"*rank*3, "Line 206 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank, "ib", ib)
                    optimizer.zero_grad()
                    # print("----"*rank*3, "Line 208 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank, "ib", ib)
                    y_hat = ddp_model(batch_x, max_len=num_vocab).to(rank) # , batch_y
                    # print("----"*rank*3, "Line 210 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank, "ib", ib)
                    # print("Ytrue size: ", batch_y.size())
                    loss = criterion(y_hat, batch_y)
                    # print("----"*rank*3, "Line 213 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank, "ib", ib)
                    if isTrain:
                        loss.backward()
                        optimizer.step()
                    # print("----"*rank*3, "Line 217 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank, "ib", ib)
                    # loss_value = loss.item()
                    # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    # loss /= world_size
                    batch_loss[ib, rank] = loss.item()
                    # print(rank, loss.item())
                # print("----"*rank*3, "Line 223 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank)
                mesg = "~Epoch[{:03d}]~ ".format(idx_epoch+1)
                mesg = mesg+" at Rank {:01d} ".format(rank)
                mesg = mesg+iter_tag+" [{:03d}]/[{:03d}]:".format(idx_file_group*4+rank+1, total_file)
                mesg = mesg+"-> Loss: "+str(np.mean(batch_loss))
                print(mesg)
                case_loss[idx_file_group * 4 + rank] = np.mean(batch_loss[:, rank])
                # print("----"*rank*3, "Line 230 at Rank ", rank, "with epoch ", idx_file_group * 4 + rank)

            # dist.barrier()
            if rank == 0:
                mesg = "~Epoch[{:03d}]~ ".format(idx_epoch+1)
                mesg = mesg+"-> Loss: "+str(np.mean(case_loss))
                print(mesg)
                np.save(train_dict["save_folder"]+"loss/epoch_loss_"+iter_tag+"_{:03d}_rank{:01d}.npy".format(idx_epoch+1, rank), case_loss)
                torch.save(ddp_model, train_dict["save_folder"]+"Ep{:03d}_model_best_ddp_Loss_{:6e}.pth".format(idx_epoch+1, np.mean(case_loss)))

            if isVal:
                if np.mean(case_loss) < best_val_loss:
                    # save the best model
                    torch.save(model, train_dict["save_folder"]+"model_best_ddp_{:03d}_rank{}.pth".format(idx_epoch+1, rank))
                    print("Checkpoint saved at Epoch {:03d}".format(idx_epoch + 1))
                    best_val_loss = np.mean(case_loss)

    cleanup()


if __name__ == "__main__":



    # ==================== DDP setting ====================

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # ==================== dict and config ====================

    # train_dict = {}
    # train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # train_dict["project_name"] = "CTGM_2d_v2_mse_102_ddp"
    # train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"
    # train_dict["seed"] = 426
    # train_dict["input_size"] = [256, 256]
    # ax, ay = train_dict["input_size"]
    # train_dict["gpu_ids"] = [1,3,4,6]
    # train_dict["epochs"] = 600
    # train_dict["batch"] = 16 * 4
    # train_dict["dropout"] = 0
    # train_dict["model_term"] = "ComplexTransformerGenerationModel"

    # train_dict["model_related"] = {}
    # train_dict["model_related"]["cx"] = 32
    # cx = train_dict["model_related"]["cx"]
    # train_dict["model_related"]["input_dims"] = [cx**2, cx**2]
    # train_dict["model_related"]["hidden_size"] = 1024
    # train_dict["model_related"]["embed_dim"] = 1024
    # train_dict["model_related"]["output_dim"] = cx**2*2
    # train_dict["model_related"]["num_heads"] = cx
    # train_dict["model_related"]["attn_dropout"] = 0.0
    # train_dict["model_related"]["relu_dropout"] = 0.0
    # train_dict["model_related"]["res_dropout"] = 0.0
    # train_dict["model_related"]["out_dropout"] = 0.0
    # train_dict["model_related"]["layers"] = 6
    # train_dict["model_related"]["attn_mask"] = False

    # train_dict["folder_X"] = "./data_dir/Iman_MR/kspace_2d/"
    # train_dict["folder_Y"] = "./data_dir/Iman_CT/kspace_2d/"
    # # train_dict["pre_train"] = "swin_base_patch244_window1677_kinetics400_22k.pth"
    # train_dict["val_ratio"] = 0.3
    # train_dict["test_ratio"] = 0.2

    # train_dict["loss_term"] = "MSELoss"
    # train_dict["optimizer"] = "AdamW"
    # train_dict["opt_lr"] = 1e-3 # default
    # train_dict["opt_betas"] = (0.9, 0.999) # default
    # train_dict["opt_eps"] = 1e-8 # default
    # train_dict["opt_weight_decay"] = 0.01 # default
    # train_dict["amsgrad"] = False # default

    for path in [train_dict["save_folder"], train_dict["save_folder"]+"npy/", train_dict["save_folder"]+"loss/"]:
        if not os.path.exists(path):
            os.mkdir(path)

    np.save(train_dict["save_folder"]+"dict.npy", train_dict)
    np.random.seed(train_dict["seed"])

    # ==================== data division ====================

    # X_list = sorted(glob.glob(train_dict["folder_X"]+"*.npy"))
    # Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.npy"))

    # selected_list = np.asarray(X_list)
    # np.random.shuffle(selected_list)
    # selected_list = list(selected_list)

    # val_list = selected_list[:int(len(selected_list)*train_dict["val_ratio"])]
    # val_list.sort()
    # test_list = selected_list[-int(len(selected_list)*train_dict["test_ratio"]):]
    # test_list.sort()
    # train_list = list(set(selected_list) - set(val_list) - set(test_list))
    # train_list.sort()

    # data_division_dict = {
    #     "train_list_X" : train_list,
    #     "val_list_X" : val_list,
    #     "test_list_X" : test_list}
    np.save(train_dict["save_folder"]+"data_division.npy", data_division_dict)


    # ==================== basic settings ====================

    # gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # local_rank = int(os.environ["LOCAL_RANK"])
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    # device = torch.device("cuda", local_rank, world_size=4)
    # print("Local rank:", local_rank)

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


    # ==================== DDP training ====================
    # initialize the process group
    gpu_list = ','.join(str(x) for x in train_dict["gpu_ids"])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    os.environ['NCCL_DEBUG'] = "INFO"
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = len(train_dict["gpu_ids"])
    # dist.init_process_group("nccl", rank=world_size, world_size=world_size)
    run_demo(demo_basic, world_size)




