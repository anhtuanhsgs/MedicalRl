# import cv2
from scipy.ndimage.interpolation    import map_coordinates
# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation
from skimage import io
from scipy.ndimage.filters import gaussian_filter as gaussian
import time
from scipy.ndimage.morphology import binary_erosion
from skimage.segmentation import find_boundaries

import matplotlib.pyplot as plt


from ZebrDataset import *

INPUT_SHAPE = (1, 64, 64, 64)

def read_im (paths, downsample=1):
    ret = []
    for path in paths:
        ret.append (io.imread (path)[::downsample,::downsample,::downsample])
    return ret

def get_data ():
    base_path = '../DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path, downsample=1) [0]
    y_train = read_im (train_label_path, downsample=1) [0]
    return X_train, y_train

if __name__ == '__main__':
    X_train, y_train = get_data ()
    zebrafish_data = ZebrDataset ('train', X_train, y_train, size=INPUT_SHAPE, device=None)
    train_data = DataLoader (zebrafish_data, batch_size=1, shuffle=True, num_workers=3)

    for sample in train_data:
        raw = sample ['raw']
        target = sample ['lbl']
        raw = raw.detach ().cpu ().numpy() [0, 0]
        target = target.detach ().cpu ().numpy ()[0, 0]
        print (raw.shape, target.shape)
        z_range, y_range, x_range = raw.shape
        z, y, x = z_range // 2, y_range // 2, x_range // 2
        yx_raw = raw [z,:,:]
        zx_raw = raw [:,y,:]
        yx_lbl = target [z,:,:]
        zx_lbl = target [:,y,:]

        yx_log_img = np.concatenate ([yx_raw, yx_lbl], 0)
        zx_log_img = np.concatenate ([zx_raw, zx_lbl], 0)
        log_img = np.concatenate ([yx_log_img, zx_log_img], 1)

        io.imsave ('sample_raw.tif', raw)
        io.imsave ('sample_lbl.tif', target)
        plt.imshow (log_img, cmap='gray')
        plt.show ()