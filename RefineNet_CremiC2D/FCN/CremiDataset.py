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

from img_aug_func import *

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed


class CremiDataset (Dataset):
    def __init__ (self, set_type, X, y, sample_size):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = sample_size
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())

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

        raw_patch = np.repeat (np.expand_dims (raw_patch, 0), 2, axis=0)
        lbl_patch = np.expand_dims (lbl_patch, 0)

        return  {'raw': raw_patch, 'lbl': lbl_patch}

class CremiDatasetRefine (Dataset):
    def __init__ (self, set_type, X, y, sample_size, pred_func, refine_rate, refine_size, refine_factor, patch_factor):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = sample_size
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())
        self.pred_func = pred_func
        self.refine_rate = refine_rate
        self.refine_size = refine_size
        self.refine_factor = refine_factor
        self.patch_factor = patch_factor

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
        size = self.refine_size
        z0 = self.rng.randint (0, len (self.raw_list))
        y0 = self.rng.randint (0, self.raw_list.shape[1] - size[0] + 1)
        x0 = self.rng.randint (0, self.raw_list.shape[2] - size[1] + 1)

        #Full resolution extraction of higher layer
        raw_patch = np.copy (self.raw_list [z0, y0:y0+size[0], x0:x0 + size[1]])
        lbl_patch = np.copy (self.lbl_list [z0, y0:y0+size[0], x0:x0 + size[1]])

        raw_patch, lbl_patch = self.aug ([raw_patch, lbl_patch])        

        raw_patch = np.repeat (np.expand_dims (raw_patch, 0), 2, axis=0)

        #Prediction of higher layer
        if (self.rng.rand () <= refine_rate):
            mask = self.pred_func (raw_patch, self.refine_factor)
            raw_patch [1:2, :, :] = np.expand_dims (mask, 0)

        #Down sample of higher layer
        raw_patch = raw_patch [::, ::patch_factor, ::patch_factor]
        lbl_patch = np.expand_dims (lbl_patch, 0) [::, ::patch_factor, ::patch_factor]

        extract_id = self.rng.randint (0, 5)
        
        sample_size = self.sample_size
        dy = [0, 0, 1, 1, 0.5]
        dx = [0, 1, 0, 1, 0.5]

        y1 = int (dy[extract_id] * sample_size[0])
        x1 = int (dx[extract_id] * sample_size[1])
        raw_patch = raw_patch [::, y1:y1 + sample_size[0], x1:x1 + sample_size[1]]
        lbl_patch = lbl_patch [::, y1:y1 + sample_size[0], x1:x1 + sample_size[1]]
        return  {'raw': raw_patch, 'lbl': lbl_patch}



