import numpy as np
from scipy.stats import mode

model_group = [
    [1001, [0],],
    [1002, [0],],
    [1003, [0],],
    [1004, [0],],
    [1005, [2],],
    [1006, [2],],
    [1007, [2],],
    [1008, [2],],
    [1009, [3],],
    [1010, [7],],
    [1011, [3],],
    [1012, [7],],
    [1013, [3],],
    [1013, [3],],
    [1014, [3],],
    [1015, [0],],
    [1016, [0],],
    [1017, [7],],
    [1018, [7],],
    [1019, [2],],
    [1020, [2],],
    [1021, [3],],
    [1022, [3],],
    [1023, [0],],
    [1024, [0],],
    [1025, [7],],
    [1026, [7],],
    [1027, [7],],
    [1028, [2],],
    [1029, [3],],
    [1030, [3],],
    [1031, [3],],
    [1032, [3],],
]

img_group = [
    ["img0026_z_RAS_1.5_1.5_2.0_vote.npy", (266, 213, 326)],
    ["img0027_z_RAS_1.5_1.5_2.0_vote.npy", (261, 216, 218)],
    ["img0028_z_RAS_1.5_1.5_2.0_vote.npy", (259, 203, 221)],
    ["img0029_z_RAS_1.5_1.5_2.0_vote.npy", (296, 230, 150)],
    ["img0030_z_RAS_1.5_1.5_2.0_vote.npy", (246, 189, 229)],
    ["img0031_z_RAS_1.5_1.5_2.0_vote.npy", (256, 197, 139)],
]

n_model = len(model_group)
n_img = len(img_group)

for idx_i in range(n_img):
    img_collection = np.zeros(
        (
            n_model, 
            img_group[idx_i][1][0],
            img_group[idx_i][1][1],
            img_group[idx_i][1][2]
        )
    )
    for idx_m in range(n_model):
        file_name = "./project_dir/Seg532_Unet_seed"+str(model_group[idx_m][0])+"/"+img_group[idx_i][0]
        print(file_name)
        file_data = np.load(file_name)
        img_collection[idx_m, :, :, :] = file_data

    img_mode = np.squeeze(mode(img_collection, axis=0).mode)
    for idx_diff in range(n_model):
        img_collection[idx_diff, :, :, :] = np.squeeze(img_collection[idx_diff, :, :, :] - img_mode)
    img_collection = np.abs(img_collection)
    img_collection[img_collection>0] = 1

    img_pct = np.sum(img_collection, axis=0)/n_model

    np.save(
        "./project_dir/Seg532_Unet_seed"+str(model_group[idx_m][0])+"/"+img_group[idx_i][0].replace("z", "e"),
        img_mode,
    )
    print("./project_dir/Seg532_Unet_seed"+str(model_group[idx_m][0])+"/"+img_group[idx_i][0].replace("z", "e"))

    np.save(
        "./project_dir/Seg532_Unet_seed"+str(model_group[idx_m][0])+"/"+img_group[idx_i][0].replace("z", "p"),
        img_pct,
    )
    print("./project_dir/Seg532_Unet_seed"+str(model_group[idx_m][0])+"/"+img_group[idx_i][0].replace("z", "p"))
