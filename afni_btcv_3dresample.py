import os
import glob

# x_list = [
#     "img0035.nii.gz",
#     "img0036.nii.gz",
#     "img0037.nii.gz",
#     "img0038.nii.gz",
#     "img0039.nii.gz",
#     "img0040.nii.gz",
# ]

# y_list = [
#     "label0035.nii.gz",
#     "label0036.nii.gz",
#     "label0037.nii.gz",
#     "label0038.nii.gz",
#     "label0039.nii.gz",
#     "label0040.nii.gz",
# ]

x_path = "./data_dir/JN_BTCV/imagesTr_re/"
y_path = "./data_dir/JN_BTCV/labelsTr_re/"

x_list = sorted(glob.glob(x_path+"*.nii.gz"))
y_list = sorted(glob.glob(y_path+"*.nii.gz"))

for x_name in x_list:
    print(x_name)
    cmd = "3dresample -input "+x_name+" -rmode Cu -dxyz 1.5, 1.5, 2.0 -orient RAS "
    cmd += "-prefix "+x_name.replace(".nii.gz", "_re.nii.gz")
    print(cmd)
    os.system(cmd)

for y_name in y_list:
    print(y_name)
    cmd = "3dresample -input "+y_name+" -rmode Cu -dxyz 1.5, 1.5, 2.0 -orient RAS "
    cmd += "-prefix "+y_name.replace(".nii.gz", "_re.nii.gz")
    print(cmd)
    os.system(cmd)
