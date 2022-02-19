import numpy as np
import matplotlib.pyplot as plt
import glob
import copy

n_epoch = 100
folder = "./project_dir/Swin3d_Iman/"

stage_hub = []
npy_list = sorted(glob.glob(folder+"loss/epoch_loss_*.npy"))
for npy_path in npy_list:
    print(npy_path)
    stage_name = npy_path[-13:-8]
    print(stage_name)
    if not model_name in model_hub:
        model_hub.append(model_name)
# print(model_hub)

# model_hub = ['CT_t', 'CT_v', 'MR_t', 'MR_v', 'naive-tf_t', 'naive-tf_v', 'naive_t', 'naive_v', 'naive_skip_t', 'naive_skip_v']

loss = np.zeros((n_epoch))
plot_target = []
for model_name in model_hub:
    current_package = [model_name]
    for idx in range(n_epoch):
        num = "{:03d}".format(idx+1)
        name = "./bridge_3000/{}/loss/epoch_loss_{}_{}.npy".format(model_name[:-2], model_name[-1], num)
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
plt.title("Training curve")

plt.savefig("./bridge_3000/loss_{}.jpg".format(n_epoch))