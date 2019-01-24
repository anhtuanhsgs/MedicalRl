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
    def __init__ (self, X, y):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = (256, 256)
        self.out_size = (256, 256)
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
        lbl_patch = np.expand_dims (lbl_patch, 0)
        raw_patch = np.expand_dims (raw_patch, 0)

        return  {'raw': raw_patch, 'lbl': lbl_patch}
