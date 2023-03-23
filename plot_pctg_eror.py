import numpy as np

def create_pctg_eror_stat(ptcg_flatten, eror_flatten):
    # Find the unique values and their counts in each array
    unique_errors = np.unique(eror_flatten)
    unique_pctgs = np.unique(ptcg_flatten)
    # Create pairs of values and their corresponding counts
    pairs = np.array(np.meshgrid(unique_errors, unique_pctgs)).T.reshape(-1, 2)
    # Find the count of each pair
    pair_counts = []
    for pair in pairs:
        indices = np.where((eror_flatten == pair[0]) & (ptcg_flatten == pair[1]))
        count = len(indices[0])
        pair_counts.append(count)
    # Print the pairs and their corresponding counts
    # for i, pair in enumerate(pairs):
    #     print("Pair {}: {} - Count: {}".format(i+1, pair, pair_counts[i]))
    # Reshape the counts into a 2D array
    seg_heatmap = np.asarray(pair_counts).reshape((len(unique_errors), len(unique_pctgs)))
    # normalize seg_heatmap with axis = 0
    # seg_heatmap = seg_heatmap / seg_heatmap.sum(axis=0)
    pctg_eror_stat = np.zeros((3, len(unique_pctgs)))
    for i in range(len(unique_pctgs)):
        pctg_eror_stat[0, i] = unique_pctgs[i]
        pctg_eror_stat[1, i] = seg_heatmap[0, i]
        pctg_eror_stat[2, i] = seg_heatmap[1, i]
    return pctg_eror_stat

model_name_list = [
    # "Seg532_basic_ensemble",
    "Seg532_Unet_ab2",
    "Seg532_Unet_ab4",
    "Seg532_Unet_ab1114444",
    "Seg532_Unet_ab4444111",
    "Seg532_UnetR_ab2",
    "Seg532_UnetR_ab2444444444",
    "Seg532_UnetR_ab41111111111",
]

case_name_list = [
"img0026",
"img0027",
"img0028",
"img0029",
"img0030",
"img0031",
]

pctg_affix = "_pct_RAS_1.5_1.5_2.0.npy"
pred_affix = "_z_RAS_1.5_1.5_2.0.npy"
folder_name = "results/moved_files_Nov5"

for model_name in model_name_list:
    pctg_eror_stat = dict()
    for case_name in case_name_list:
        pctg_name = "./" + folder_name + "/" + model_name + "_" + case_name + pctg_affix
        pred_name = "./" + folder_name + "/" + model_name + "_" + case_name + pred_affix
        grth_name = "./" + folder_name + "/" + "GT" + "_" +case_name + "_z_RAS_1.5_1.5_2.0.npy"
        pctg_data = np.load(pctg_name)
        pred_data = np.load(pred_name)
        grth_data = np.load(grth_name)
        print(pred_name)
        ptcg_flatten = (pctg_data).flatten()
        eror_data = np.abs(pred_data - grth_data)
        eror_data[eror_data>0] = 1
        eror_flatten = eror_data.flatten() 
        curr_pctg_eror = create_pctg_eror_stat(ptcg_flatten, eror_flatten)
        for i in range(curr_pctg_eror.shape[1]):
            keyname = str(curr_pctg_eror[0, i])
            if keyname in pctg_eror_stat:
                pctg_eror_stat[keyname].append([curr_pctg_eror[1, i], curr_pctg_eror[2, i]])
            else:
                pctg_eror_stat[keyname] = [curr_pctg_eror[1, i], curr_pctg_eror[2, i]]
    pctg_eror_plot = np.zeros((2, len(pctg_eror_stat.keys())))
    for idx, key in enumerate(pctg_eror_stat.keys()):
        pctg_eror_plot[0, idx] = key
        if len(pctg_eror_stat[key]) == 1:
            pctg_total = np.array(pctg_eror_stat[key])
        else:
            pctg_total = np.sum(np.array(pctg_eror_stat[key]), axis=0)
        print(pctg_total)
        pctg_eror_plot[1, idx] = pctg_total[1] / np.sum(pctg_total)
    save_name = "./" + folder_name + "/stat_" + model_name + "_pctg_eror.npy"
    np.save(save_name, pctg_eror_plot)
    print("Saved to " + save_name)
    