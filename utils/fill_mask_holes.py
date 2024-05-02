import numpy as np
from skimage.morphology import binary_fill_holes

def fill_binary_holes(mask_data):
    ax, ay, az = mask_data.shape
    # iterate over slices from the three dims
    for i in range(ax):
        slice = mask_data[i, :, :]
        slice = binary_fill_holes(slice).astype(int)
        mask_data[i, :, :] = slice

    for i in range(ay):
        slice = mask_data[:, i, :]
        slice = binary_fill_holes(slice).astype(int)
        mask_data[:, i, :] = slice

    for i in range(az):
        slice = mask_data[:, :, i]
        slice = binary_fill_holes(slice).astype(int)
        mask_data[:, :, i] = slice
    
    for idz in range(az):
        # we find the idx of first non-zero value, use the idx+1 mask slice to replace the current mask slice
        if np.mean(mask_data[:, :, idz]) > 0:
            mask_data[:, :, idz] = mask_data[:, :, idz+1]
            # only do it once and break
            break
    

    return mask_data