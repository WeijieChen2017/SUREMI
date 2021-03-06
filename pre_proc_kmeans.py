import os
import glob
import time
import numpy as np
import nibabel as nib
from sklearn import cluster


def bin_CT(img, n_bin=128):
    data_vector = img
    data_max = np.amax(data_vector)
    data_min = np.amin(data_vector)
    data_squeezed = (data_vector-data_min)/(data_max-data_min)
    data_extended = data_squeezed * (n_bin-1)
    data_discrete = data_extended // 1
    return np.asarray(list(data_discrete), dtype=np.int64)


def generate_dist_weights(data_shape):
    dist = np.zeros(data_shape)
    len_x = data_shape[0]
    len_y = data_shape[1]
    len_z = data_shape[2]
    center = [len_x // 2, len_y // 2, len_z // 2]
    for ix in range(len_x):
        for iy in range(len_y):
            for iz in range(len_z):
                dx = np.abs(ix-center[0]) ** 2
                dy = np.abs(iy-center[1]) ** 2
                dz = np.abs(iz-center[2]) ** 2
                dist[ix, iy, iz] = np.sqrt(dx+dy+dz)
    
    return dist



def dist_kmeans(X_path, nX_clusters, dist):
    X_file = nib.load(X_path)
    X_data = bin_CT(X_file.get_fdata(), n_bin=n_bin)
    
    X_cluster = cluster.KMeans(n_clusters=nX_clusters)
    X_flatten = np.ravel(X_data)
    X_flatten = np.reshape(X_flatten, (len(X_flatten), 1))
    X_flatten_k = X_cluster.fit_predict(X_flatten)
    X_data_k = np.reshape(X_flatten_k, X_data.shape)
    
    weight_data = np.multiply(X_data_k, dist)
    scores = np.zeros((nX_clusters))
    for idx in range(nX_clusters):
        cluster_map = np.where(X_data==idx, 1, 0)
        scores[idx] = np.sum(np.multiply(cluster_map, dist)) / np.sum(cluster_map)
    idx_scores = np.argsort(scores)
    print(idx_scores, scores)
    
    for idx in range(nX_clusters):
        X_data_k[X_data_k == idx] = nX_clusters+idx
    
    for idx in range(nX_clusters):
        X_data_k[X_data_k == nX_clusters+idx] = idx_scores[idx]
    
    return X_data_k


train_dict = {}
train_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
train_dict["project_name"] = "pixel_correlation"
train_dict["save_folder"] = "./project_dir/"+train_dict["project_name"]+"/"

train_dict["folder_X"] = "./data_dir/norm_MR/regular/"
train_dict["folder_Y"] = "./data_dir/norm_CT/regular/"
train_dict["new_folder_X"] = "kmeans10"
if not os.path.exists(train_dict["folder_X"].replace("regular", train_dict["new_folder_X"])):
    os.makedirs(train_dict["folder_X"].replace("regular", train_dict["new_folder_X"]))

X_list = sorted(glob.glob(train_dict["folder_X"]+"*.nii.gz"))
Y_list = sorted(glob.glob(train_dict["folder_Y"]+"*.nii.gz"))

nX_clusters = 10
n_bin = 256
dist = generate_dist_weights(nib.load(X_list[0]).get_fdata().shape)

for cnt_file, file_path in enumerate(X_list):
    print(file_path)
    X_path = file_path
    X_file = nib.load(X_path)
    X_data_k = dist_kmeans(X_path, nX_clusters, dist)
    X_save_name = X_path.replace("regular", train_dict["new_folder_X"])
    X_save_file = nib.Nifti1Image(X_data_k, X_file.affine, X_file.header)
    nib.save(X_save_file, X_save_name)
