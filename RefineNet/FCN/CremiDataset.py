import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from natsort import natsorted
import os, sys, argparse, glob
import skimage.io as io
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.measure import label
from skimage.transform import resize
import time
import math

from img_aug_func import *

ORIGIN_SIZE = (512, 512)

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed


class CremiDataset (Dataset):
    def __init__ (self, set_type, X, y, level, ds_factor, zoom_factor):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = list (ORIGIN_SIZE)
        for i in range (level):
            self.sample_size [0] = math.ceil (self.sample_size [0] * zoom_factor)
            self.sample_size [1] = math.ceil (self.sample_size [1] * zoom_factor) 
        self.out_size = (math.ceil (self.sample_size[0] * ds_factor), 
                        math.ceil (self.sample_size[1] * ds_factor))
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())
        self.ds_factor = ds_factor

    def __len__ (self):
        return self.iter_per_epoch * 200

    def aug (self, imgs):
        ret = []
        rotk = self.rng.randint (0, 4)
        flipk = self.rng.randint (1, 5)
        for img in imgs:
            img = np.rot90(img, rotk, axes=(0,1))
            img = random_flip (img, flipk)
            ret += [img.copy ()]
        return ret

    def __getitem__ (self, idx):
        size = self.sample_size
        z0 = self.rng.randint (0, len (self.raw_list))
        y0 = self.rng.randint (0, self.raw_list.shape[1] - size[0] + 1)
        x0 = self.rng.randint (0, self.raw_list.shape[2] - size[1] + 1)

        raw_patch = np.copy (self.raw_list [z0, y0:y0+size[0], x0:x0 + size[1]])
        lbl_patch = np.copy (self.lbl_list [z0, y0:y0+size[0], x0:x0 + size[1]])

        raw_patch, lbl_patch = self.aug ([raw_patch, lbl_patch])

        raw_patch = resize (raw_patch, self.out_size, order=0, mode='wrap', preserve_range=True) #Back to origin size   
        lbl_patch = resize (lbl_patch, self.out_size, order=0, mode='wrap', preserve_range=True) #Back to origin size   

        raw_patch = np.repeat (np.expand_dims (raw_patch, 0), 2, axis=0)
        lbl_patch = np.expand_dims (lbl_patch, 0)

        return  {'raw': raw_patch, 'lbl': lbl_patch}



class CremiDatasetRefine (Dataset):
    def __init__ (self, set_type, X, y, level, ds_factor, zoom_factor, uplevel_model, ds_factor_ulv, device):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = list (ORIGIN_SIZE)
        self.ulv_size = list (ORIGIN_SIZE)

        for i in range (level - 1):
            self.ulv_size [0] = math.ceil (self.ulv_size [0] * zoom_factor)
            self.ulv_size [1] = math.ceil (self.ulv_size [1] * zoom_factor) 

        for i in range (level):
            self.sample_size [0] = math.ceil (self.sample_size [0] * zoom_factor)
            self.sample_size [1] = math.ceil (self.sample_size [1] * zoom_factor)

        self.out_size = (math.ceil (self.sample_size[0] * ds_factor), 
                        math.ceil (self.sample_size[1] * ds_factor))

        self.ulv_in_size = (math.ceil (self.ulv_size [0] * ds_factor_ulv),
                            math.ceil (self.ulv_size [1] * ds_factor_ulv))
        
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())
        self.ds_factor = ds_factor
        self.ulv_model = uplevel_model
        self.device = device

    def __len__ (self):
        return self.iter_per_epoch * 200

    def aug (self, imgs):
        ret = []
        rotk = self.rng.randint (0, 4)
        flipk = self.rng.randint (1, 5)
        for img in imgs:
            img = np.rot90(img, rotk, axes=(0,1))
            img = random_flip (img, flipk)
            ret += [img.copy ()]
        return ret

    def __getitem__ (self, idx):
        ulv_size = self.ulv_size
        z0 = self.rng.randint (0, len (self.raw_list))
        y0 = self.rng.randint (0, self.raw_list.shape[1] - ulv_size[0] + 1)
        x0 = self.rng.randint (0, self.raw_list.shape[2] - ulv_size[1] + 1)

        raw_patch_ulv = np.copy (self.raw_list [z0, y0:y0+ulv_size[0], x0:x0 + ulv_size[1]])
        lbl_patch_ulv = np.copy (self.lbl_list [z0, y0:y0+ulv_size[0], x0:x0 + ulv_size[1]])
        raw_patch_ulv, lbl_patch_ulv = self.aug ([raw_patch_ulv, lbl_patch_ulv])

        use_ulv_mask = self.rng.rand () > 0.3;
        if use_ulv_mask:
            # Resize to up level model input
            raw_patch_ulv_ds = resize (raw_patch_ulv, self.ulv_in_size, order=0, mode='wrap', preserve_range=True)
            raw_patch_ulv_t = torch.tensor (raw_patch_ulv_ds, device=self.device, dtype=torch.float32, requires_grad=False) / 255.0
            raw_patch_ulv_t = raw_patch_ulv_t.unsqueeze (0).unsqueeze (0)
            raw_patch_ulv_t = raw_patch_ulv_t.repeat (1, 2, 1, 1)
            mask_patch_ulv = self.ulv_model (raw_patch_ulv_t).detach ().cpu ().numpy ()[0,0,:,:]
            # Refine prediction back to 1/1 data size
            mask_patch_ulv = resize (mask_patch_ulv, ulv_size, order=0, mode='wrap', preserve_range=True)
            mask_patch_ulv = (mask_patch_ulv * 255).astype (np.uint8)
        else:
            mask_patch_ulv = np.zeros_like (raw_patch_ulv)

        sample_size = self.sample_size

        y0 = self.rng.randint (0, ulv_size[0] - sample_size[0] + 1)
        x0 = self.rng.randint (0, ulv_size[1] - sample_size[1] + 1)

        raw_patch = raw_patch_ulv [y0:y0+sample_size[0], x0:x0 + sample_size[1]]
        lbl_patch = lbl_patch_ulv [y0:y0+sample_size[0], x0:x0 + sample_size[1]]
        mask_patch = mask_patch_ulv [y0:y0+sample_size[0], x0:x0 + sample_size[1]]

        if self.rng.rand () > 0.5:
            raw_patch = np.zeros_like (raw_patch)

        raw_patch = np.expand_dims (resize (raw_patch, self.out_size, order=0, mode='wrap', preserve_range=True), 0)
        mask_patch = np.expand_dims (resize (mask_patch, self.out_size, order=0, mode='wrap', preserve_range=True), 0)
        lbl_patch = resize (lbl_patch, self.out_size, order=0, mode='wrap', preserve_range=True)    

        raw_patch = np.concatenate ([raw_patch, mask_patch], 0)
        lbl_patch = np.expand_dims (lbl_patch, 0)

        return  {'raw': raw_patch, 'lbl': lbl_patch}



