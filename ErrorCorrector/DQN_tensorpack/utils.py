import numpy as np
import os,sys, argparse, glob
sys.path.append('../')
from skimage.measure import label
from Utils.img_aug_func import *
from Utils.utils import *
from natsort import natsorted


def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train

def setup_data (env_conf):
    raw , gt_lbl = get_data (path='../Data/train/', args=None)
    prob = io.imread ('../Data/train-membranes-idsia.tif')
    lbl = []
    for img in prob:
        lbl += [label (img > env_conf ['cell_thres'])]
    lbl = np.array (lbl)
    return raw, lbl, prob, gt_lbl
