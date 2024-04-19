# load two masks and plot them side by side
import os
import glob
import numpy as np
import nibabel as nib

boundary_list = [10, 15, 20, 25]
data_folder = "results/dice_iou/"
case_list = sorted(glob.glob(f"{data_folder}/*/*mask*.nii.gz"))
model_id_list = []
case_id_list = []
for case in case_list:
    model_id = case.split("/")[2]
    if model_id not in model_id_list:
        model_id_list.append(model_id)
    case_id = case.split("/")[3].split("_")[0]
    if case_id not in case_id_list:
        case_id_list.append(case_id)
print(model_id_list)
print(case_id_list)