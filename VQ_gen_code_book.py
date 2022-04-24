import numpy as np
from scipy.cluster import vq
from  sklearn.cluster import MiniBatchKMeans

def dist_l2(x, y):
    return np.sqrt(np.sum(np.square(x-y)))

array_x_patch = np.load("./data_dir/Iman_MR/VQ_8x/array_x_patch.npy")
array_y_patch = np.load("./data_dir/Iman_CT/VQ_8x/array_y_patch.npy")

cnt_samples, cnt_feature = array_x_patch.shape

# whitening by features divided by the std

std_x = np.zeros((cnt_feature))
std_y = np.zeros((cnt_feature))

for idx in range(cnt_feature):
    std_x[idx] = np.std(array_x_patch[:, idx])
    std_y[idx] = np.std(array_y_patch[:, idx])

save_name_x = "./data_dir/Iman_MR/VQ_8x/std_x_patch.npy"
save_name_y = "./data_dir/Iman_CT/VQ_8x/std_y_patch.npy"
np.save(save_name_x, std_x)
np.save(save_name_y, std_y)
print(save_name_x, save_name_y)

whitened_X = array_x_patch / std_x
whitened_Y = array_y_patch / std_y

print(array_x_patch[0, :], whitened_X[0, :])

MBK_X = MiniBatchKMeans(n_clusters=4096, random_state=426, batch_size=1024, verbose=1)
MBK_Y = MiniBatchKMeans(n_clusters=4096, random_state=426, batch_size=1024, verbose=1)

MBK_X.fit(whitened_X)
MBK_Y.fit(whitened_Y)

save_name_x = "./data_dir/Iman_MR/VQ_8x/MBK_x_patch.npy"
save_name_y = "./data_dir/Iman_CT/VQ_8x/MBK_y_patch.npy"
np.save(save_name_x, MBK_X)
np.save(save_name_y, MBK_Y)
print(save_name_x, save_name_y)


cluster_centers_X = MBK_X.cluster_centers_
cluster_centers_Y = MBK_Y.cluster_centers_

label_X = MBK_X.predict(whitened_X)
label_Y = MBK_Y.predict(whitened_Y)

l2_sum_X = 0
l2_sum_Y = 0
for idx in range(cnt_samples):
    l2_sum_X += dist_l2(whitened_X[idx, :], cluster_centers_X[label_X[idx], :])
    l2_sum_Y += dist_l2(whitened_Y[idx, :], cluster_centers_Y[label_Y[idx], :])
print(l2_sum_X/cnt_samples, l2_sum_Y/cnt_samples)
