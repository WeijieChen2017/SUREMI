import numpy as np
import matplotlib.pyplot as plt
import glob
import copy
import os

n_epoch = 391
folder = "./project_dir/syn_DLE_4444444_e400/"

stage_hub = []
npy_list = sorted(glob.glob(folder+"loss/epoch_loss_*.npy"))
for npy_path in npy_list:
    print(npy_path)
    stage_name = os.path.basename(npy_path)
    stage_name = stage_name.split("_")[2]
    print(stage_name)
    if not stage_name in stage_hub:
        stage_hub.append(stage_name)

loss = np.zeros((n_epoch))
plot_target = []
for stage_name in stage_hub:
    current_package = [stage_name]
    for idx in range(n_epoch):
        num = "{:03d}".format(idx+1)
        name = folder+"loss/epoch_loss_{}_{}.npy".format(stage_name, num)
        data = np.load(name)
        loss[idx] = np.mean(data)
    current_package.append(copy.deepcopy(loss))
    plot_target.append(current_package)

legend_list = []
plt.figure(figsize=(9,6), dpi=300)
for package in plot_target:
    loss_array = package[1]
    loss_tag = package[0]
    legend_list.append(loss_tag)
    print(loss_tag, np.mean(loss_array))
    plt.plot(range(n_epoch), loss_array)

plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")
plt.legend(legend_list)
plt.title("Training curve of "+folder.split("/")[-2])

plt.savefig(folder + "loss_{}.jpg".format(n_epoch))