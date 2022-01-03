import time
import numpy as np

pre_proc_dict = {}

pre_proc_dict["dir_x_orig"] = 
pre_proc_dict["dir_y_orig"] = 
pre_proc_dict["dir_x_syn"] = 
pre_proc_dict["dir_y_syn"] = 
pre_proc_dict["range_norm_x"] = 
pre_proc_dict["range_norm_y"] = 
pre_proc_dict["is_seg_x"] = 
pre_proc_dict["is_seg_y"] = 
pre_proc_dict["time_stamp"] = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

np.save("./log_dir/log_pre_proc_"+pre_proc_dict["time_stamp"]+".npy", )
        