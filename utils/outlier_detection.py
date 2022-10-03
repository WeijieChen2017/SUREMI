import numpy as np

def HU_to_label(data):
    label = np.zeros(data.shape)
    mask_air = data < -500
    mask_bone = data > 500
    mask_soft_1 = data > -500
    mask_soft_2 = data < 500
    mask_soft_1 = mask_soft_1.astype(int)
    mask_soft_2 = mask_soft_2.astype(int)
    mask_soft = mask_soft_1 * mask_soft_2
    mask_soft = mask_soft.astype(bool)
    label[mask_air] = 0
    label[mask_soft] = 1
    label[mask_bone] = 2
    
    return label


def find_label_diff(data_pred, data_std):
    
	# data_pred is the predictions rangeing from [0,1]
	# data_std is the std among all predictions, empirically ranging from [0, 0.5]

    data_pred_HU = data_pred * 4000 - 1000
    data_std_HU = data_std * 4000

    data_max = data_pred_HU + data_std_HU
    data_min = data_pred_HU - data_std_HU
    
    label_max = HU_to_label(data_max)
    label_min = HU_to_label(data_min)
    label_mid = HU_to_label(data_pred_HU)
    
    label_diff_1 = label_max != label_mid
#     label_diff_1 = label_diff_1.astype(int)
    label_diff_2 = label_min != label_mid
#     label_diff_2 = label_diff_2.astype(int)
    label_diff = label_diff_1 | label_diff_2
    label_diff = label_diff.astype(int)

    # test_file = nib.Nifti1Image(np.squeeze(label_diff), file_pred.affine, file_pred.header)
    # test_save_name = os.path.dirname(path_to_pred)+"/"+case_num+"_label_diff.nii.gz"
    # nib.save(test_file, test_save_name)
    # print(test_save_name)
    
    return label_diff