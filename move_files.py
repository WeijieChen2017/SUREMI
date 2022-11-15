import numpy as np
import glob
import os

target_folder = [
"Seg532_Unet",
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
"Seg532_Unet_ab1114444",
"Seg532_Unet_ab2",
"Seg532_Unet_ab4",
"Seg532_Unet_ab4444111",
"Seg532_Unet_channnel_r050",
"Seg532_Unet_channnel_r050w",
"Seg532_Unet_channnel_r100",
"Seg532_Unet_channnel_r100w",
]

target_file = [
"img0026*RAS*",
"img0027*RAS*",
"img0028*RAS*",
"img0029*RAS*",
"img0030*RAS*",
"img0031*RAS*",
]

N_folder = len(target_folder)
N_file = len(target_file)

new_folder = "./moved_files_Nov5/"
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

for idx_folder in range(N_folder):
	for idx_file in range(N_file):
		path_src = "./project_dir/"+target_folder[idx_folder]+"/"+target_file[idx_file]
		list_src = sorted(glob.glob(path_src))
		for single_src in list_src:
			filename = os.path.basename(single_src)
			# print(single_src, filename)
			path_dst = new_folder+target_folder[idx_folder]+"_"+filename
			cmd = "cp "+ single_src + " " + path_dst
			print(cmd)
			os.system(cmd)