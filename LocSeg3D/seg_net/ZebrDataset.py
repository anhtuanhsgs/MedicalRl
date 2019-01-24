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
import os.path
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


class ZebrDataset (Dataset):
    def __init__ (self, set_type, X, y, size, device):
        print ("Initialize data flow")
        self.size = size
        self.set_type = set_type
        
        self.device = device
        self.iter_per_epoch = 0
        self.data_seed = time_seed ()
        self.rng = np.random.RandomState(self.data_seed)

        self.raw = X / 65535.0 * 255
        self.lbl = label (erode_label ([y > 0]) [0])
        self.cell_cnt = self.lbl.max ()
        self.raw = np.pad (self.raw, 80, mode='constant', constant_values=0)
        self.lbl = np.pad (self.lbl, 80, mode='constant', constant_values=0)

        print (self.raw.shape, self.lbl.shape)
        
        print ('Number of cell: ', self.cell_cnt)
        if not os.path.isfile ('bbox.npy'):
            print ("Generating bboxes")
            self.bbox = {}
            # Get boundary box for each cell
            for i in range (self.cell_cnt):
                indexes = np.where (self.lbl == i)
                self.bbox [i] = {
                                 'Z': (indexes[0].min (), indexes[0].max ()),
                                 'Y': (indexes[1].min (), indexes[1].max ()),
                                 'X': (indexes[2].min (), indexes[2].max ())
                                }

            np.save ('bbox.npy', self.bbox)
        else:
            print ("Loading bboxes")
            self.bbox = np.load ('bbox.npy').item ()

        self.iter_per_epoch = self.cell_cnt

    def __len__ (self):
        return self.iter_per_epoch // 3

    def aug (self, imgs):
        ret = []
        rotk = self.rng.randint (0, 4)
        flipk = self.rng.randint (1, 5)
        for img in imgs:
            img = square_rotate (img, rotk)
            img = random_flip (img, flipk)
            ret += [img.copy ()]
        return ret

    def __getitem__ (self, idx):
        #Get random block
        self.rng.seed (time_seed ())
        found = False
        while (not found):
            cell_id = self.rng.randint (1, self.cell_cnt)
            z = self.bbox[cell_id]['Z']
            y = self.bbox[cell_id]['Y']
            x = self.bbox[cell_id]['X']
            
            z_range = z[1] - z[0]
            y_range = y[1] - y[0]
            x_range = x[1] - x[0]
            
            if x_range <= 8 or y_range <= 8 or z_range <= 8:
                continue
            found = True

        z_crop = [z[0] - self.rng.randint (1, z_range//4*3), z[1] + self.rng.randint (1, z_range//4*3)]
        y_crop = [y[0] - self.rng.randint (1, y_range//4*3), y[1] + self.rng.randint (1, y_range//4*3)]
        x_crop = [x[0] - self.rng.randint (1, x_range//4*3), x[1] + self.rng.randint (1, x_range//4*3)]

        raw_patch = self.raw [z_crop[0]:z_crop[1]+1, y_crop[0]:y_crop[1]+1, x_crop[0]:x_crop[1]+1].astype (np.float32)
        lbl_patch = (self.lbl [z_crop[0]:z_crop[1]+1, y_crop[0]:y_crop[1]+1, x_crop[0]:x_crop[1]+1] == cell_id).astype (np.int32) * 255

        # print (lbl_patch.max (), self.lbl.max ())
        raw_patch, lbl_patch = self.aug ([raw_patch, lbl_patch])
        raw_patch = resize (raw_patch, self.size[1:], order=0, mode='reflect', preserve_range=True)
        lbl_patch = resize (lbl_patch, self.size[1:], order=0, mode='reflect', preserve_range=True) > 0
        #Torch volume format // batch_size, channel, z, x, y
        raw_patch = np.expand_dims (raw_patch, 0).astype (np.float32) 
        lbl_patch = np.expand_dims (lbl_patch, 0).astype (np.float32) * 255.0

        return  {'raw': raw_patch, 'lbl': lbl_patch}


