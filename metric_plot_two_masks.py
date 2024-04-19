# load two masks and plot them side by side
import os
import glob
import numpy as np
import nibabel as nib

data_folder = "results/dice_iou/"
case_list = sorted(glob.glob(f"{data_folder}/*/*mask*.nii.gz"))
model_id_list = []
for case in case_list:
    model_id = case.split("/")[1]
    if model_id not in model_id_list:
        model_id_list.append(model_id)
print(model_id_list)