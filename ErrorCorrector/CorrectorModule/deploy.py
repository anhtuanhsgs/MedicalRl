import os, sys, glob, time
from os import sys, path
import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import sobel

from EMDataset import EMDataset

from natsort import natsorted
sys.path.append('../')
from Utils.img_aug_func import *
from Utils.utils import *
import matplotlib.pyplot as plt
import Losses

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def np2tensor (arr):
    arr = np.expand_dims (arr, 0)
    zero_mask = np.zeros_like (arr)
    arr = np.concatenate ([arr, zero_mask], 0)
    return torch.tensor (arr [None], dtype=torch.float32).cuda ()

def build_mask_patch (shape):
    # print ("patch shape = ", shape)
    yy, xx = np.meshgrid (
            np.linspace(-1,1,shape[0], dtype=np.float32),
            np.linspace(-1,1,shape[1], dtype=np.float32)
        )
    d = np.sqrt(xx*xx+yy*yy)
    sigma, mu = 0.5, 0.0
    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    v_weight = v_weight/v_weight.max()
    return v_weight

def deploy (model, args):
    gpu_id = args.gpu_id
    data = io.imread (args.data_path)
    save_path = args.save_path + "deploy.tif"
    
    shape = data.shape [1:]
    # print ("raw shape:", shape)
    size = args.size
    
    ret = []
    for img in data:
        y0 = 0
        
        doneY = False
        mask_patch = build_mask_patch (size)
        mask = np.zeros (img.shape, dtype=np.float32)
        prob = np.zeros (img.shape, dtype=np.float32)
        while (y0 < shape [0]):
            if (y0 + size[0] >= shape[0]):
                doneY = True
                y0 = shape [0] - size[0]
            doneX = False
            x0 = 0
            while (x0 < shape [1]):
                if (x0 + size [1] >= shape [1]):
                    doneX = True
                    x0 = shape [1] - size [1]
                with torch.no_grad ():
                    with torch.cuda.device (gpu_id):
                        patch = img [y0:y0+size[0], x0:x0+size[1]]
                        # print ("abc ", patch.shape)
                        patch_t = np2tensor (patch)
                        # print ("123 ", patch_t.shape)
                        pred = model (patch_t)
                        pred = pred.cpu ().numpy () [0][0]

                        # print ("xyz ", pred.shape)
                prob [y0:y0+size[0], x0:x0+size[1]] += pred * mask_patch
                mask [y0:y0+size[0], x0:x0+size[1]] += mask_patch

                x0 += size [1] // 2
                if doneX:
                    break
            y0 += size[0] // 2
            if doneY:
                break
        prob = prob / mask
        ret.append (prob)

        # plt.imshow (mask)
        # plt.show ()
        # plt.imshow (prob)
        # plt.show ()
        # break
        
    ret = np.array (ret)
    ret = (ret * 255).astype (np.uint8)
    io.imsave (args.save_path, ret)


