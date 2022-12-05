import glob
import os
import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix
import xlsxwriter

target_folder = [
# "Seg532_Unet_ab2",
# Seg532_UnetR
# Seg532_UnetR_MC_D50_R100
# Seg532_UnetR_ab2
# Seg532_UnetR_ab2444444444
# Seg532_UnetR_ab41111111111

"Seg532_Unet_Dim3_D25_R100",
"Seg532_Unet_Dim3_D50_R050",
"Seg532_Unet_Dim3_D50_R100",
"Seg532_Unet_Dim3_D75_R100",
"Seg532_Unet_MC_D25_R100",
"Seg532_Unet_MC_D50_R001",
"Seg532_Unet_MC_D50_R010",
"Seg532_Unet_MC_D50_R100",
"Seg532_Unet_MC_D75_R100",
# "Seg532_Unet_ab1114444", "pre"
# "Seg532_Unet" 
# "Seg532_Unet_ab2", ""
# "Seg532_Unet_ab4", " pre"
# "Seg532_Unet_ab4444111", "pre"
"Seg532_Unet_channnel_r050",
"Seg532_Unet_channnel_r050w",
"Seg532_Unet_channnel_r100",
"Seg532_Unet_channnel_r100w",
]

case_list = [
"img0026",
"img0027",
"img0028",
"img0029",
"img0030",
"img0031",
]

tag = "_RAS_1.5_1.5_2.0"
n_label = 14

# first row
for idx in range(n_label):
    worksheet.write(0, idx+1, "region_"+str(idx), bold)

for idx_model, model_name in enumerate(target_folder):

    workbook = xlsxwriter.Workbook("./excel/"+model_name+"+dice.xlsx")
    # workbook = xlsxwriter.Workbook('synthesis_wilcoxon.xlsx')
    bold = workbook.add_format({'bold': True})
    set_yellow = workbook.add_format({'bg_color': 'yellow'})
    worksheet = workbook.add_worksheet()

    print(model_name)
    worksheet.write(idx_model+1, 0, model_name, bold)
    IoU = np.zeros((len(case_list), n_label))
    for idx_case, case_name in enumerate(case_list):
        # Seg532_Unet_ab2_img0026_y_RAS_1.5_1.5_2.0.npy
        y_path = "./moved_files_Nov5/"+model_name+"_"+case_name+"_y"+tag+".npy"
        z_path = "./moved_files_Nov5/"+model_name+"_"+case_name+"_z"+tag+".npy"
        # y_path = "./"+model_name+"_"+case_name+"_y"+tag+".npy"
        # z_path = "./"+model_name+"_"+case_name+"_z"+tag+".npy"
        y_data = np.load(y_path)
        z_data = np.load(z_path)
        CM = confusion_matrix(np.ravel(y_data), np.ravel(z_data))
        CM_y = np.sum(CM, axis=0)
        CM_z = np.sum(CM, axis=1)
        for i in range(n_label):
            intersection = CM[i, i]
            union = CM_y[i]+CM_z[i]
            IoU[idx_case, i] = 2*intersection/union
    IoU_mean = np.mean(IoU, axis=0)
#     IoU_std = np.std(IoU, axis=0)
    for idx in range(n_label):
        worksheet.write(idx_model+1, idx+1, "{:.4f}".format(IoU_mean[idx]))
    workbook.close()

