import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from natsort import natsorted
import os, sys, argparse, glob, copy
import skimage.io as io
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from skimage.measure import label
from skimage.transform import resize
import time
import math

import albumentations as A

ORIGIN_SIZE = (512, 512)

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ISBI2012_Dataset16 (Dataset):
    def __init__ (self, set_type, X, y, level, zoom_factor):
        self.raw_list = X
        self.lbl_list = y
        self.sample_size = list (ORIGIN_SIZE)
        for i in range (level):
            self.sample_size [0] = math.ceil (self.sample_size [0] * zoom_factor / 16) * 16
            self.sample_size [1] = math.ceil (self.sample_size [1] * zoom_factor / 16) * 16 
        self.out_size = self.sample_size

        print ('Dataset sample size:', self.sample_size)
        print ('Dataset output size:', self.out_size)
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())

    def __len__ (self):
        return self.iter_per_epoch * 100

    def aug (self, imgs):
        ret = []
        rotk = self.rng.randint (0, 4)
        flipk = self.rng.randint (1, 5)
        for img in imgs:
            img = np.rot90(img, rotk, axes=(0,1))
            img = random_flip (img, flipk)
            ret += [img.copy ()]
        return ret

    def append_concat_last (self, imgs):
        ret = []
        for img in imgs:
            ret.append (np.expand_dims (img, -1))
        return np.concatenate (ret, -1)

    def paired_aug (self, raw, lbl):
        aug = A.Compose ([
            A.HorizontalFlip (),
            A.RandomRotate90 (p=0.5),
            A.VerticalFlip (),
            A.Transpose (),
            A.ElasticTransform(p=0.75, alpha=100, sigma=240 * 0.05, alpha_affine=100 * 0.03),
            A.RandomGamma(p=1, gamma_limit=(30, 236)),
            A.RandomContrast(p=0.8),
            A.GaussNoise (p=0.5),
            A.Blur (p=0.5)
        ], p=0.8)

        ret = aug (image=raw, mask=lbl)
        raw, lbl = ret ['image'], ret['mask']
        return raw, lbl

    def __getitem__ (self, idx):
        size = self.sample_size
        z0 = self.rng.randint (0, len (self.raw_list))
        y0 = self.rng.randint (0, self.raw_list.shape[1] - size[0] + 1)
        x0 = self.rng.randint (0, self.raw_list.shape[2] - size[1] + 1)

        raw_patch = np.copy (self.raw_list [z0, y0:y0+size[0], x0:x0 + size[1]])
        lbl_patch = np.copy (self.lbl_list [z0, y0:y0+size[0], x0:x0 + size[1]])

        raw_patch, lbl_patch = self.paired_aug (raw_patch, lbl_patch)

        raw_patch = np.expand_dims (raw_patch, 0)
        mask = np.zeros_like (raw_patch)

        raw_patch = np.concatenate ([raw_patch, mask], 0)
        lbl_patch = np.expand_dims (lbl_patch, 0)

        return  {'raw': raw_patch, 'lbl': lbl_patch}

class ISBI2012_Dataset16_refine (Dataset):
    def __init__ (self, set_type, X, y, max_level, zoom_factor, models, devices):
        self.raw_list = X.astype (np.float32)
        self.lbl_list = y.astype (np.float32)
        self.devices = devices
        sample_size = list (ORIGIN_SIZE)
        self.sample_sizes = [] 
        self.max_level = max_level
        for i in range (self.max_level):
            self.sample_sizes.append (copy.deepcopy (sample_size))
            sample_size [0] = math.ceil (sample_size [0] * zoom_factor / 16) * 16
            sample_size [1] = math.ceil (sample_size [1] * zoom_factor / 16) * 16 

        print ('Dataset sample size:', self.sample_sizes)
        self.set_type = set_type
        self.iter_per_epoch = len (self.lbl_list)
        self.rng = np.random.RandomState(time_seed ())
        self.models = models
        self.max_level = max_level
        self.default_device = torch.device ("cuda")

    def __len__ (self):
        return self.iter_per_epoch * 100

    def to_tensor (self, img, level):
        # CHW img input
        # NCHW tensor ret
        with torch.cuda.device (self.devices [level]):
            img_t = torch.tensor (img, device=self.default_device, dtype=torch.float32)
            img_t = img_t.unsqueeze (0) / 255.0
            return img_t

    def concat_raw_mask (self, raw, mask):
        # HW raw, mask input
        return np.concatenate ([raw [None], mask[None]], 0)

    def append_concat_last (self, imgs):
        ret = []
        for img in imgs:
            ret.append (np.expand_dims (img, -1))
        return np.concatenate (ret, -1)

    def crop_from_prev (self, raws, lbls, masks, H, W):
        up_lv = self.rng.randint (0, len (raws))
        trippled = self.append_concat_last ([raws[up_lv], lbls[up_lv], masks[up_lv]])
        cropper = A.RandomCrop (height=H, width=W, p=1)
        cropped = cropper (image=trippled, mask=None) ['image']
        return cropped [...,0], cropped[...,1], cropped[...,2]

    def predict (self, raw, mask, level):
        with torch.no_grad ():
            with torch.cuda.device (self.devices[level]):
                paired = np.concatenate ([raw[None], mask[None]], 0)
                paired_t = self.to_tensor (paired, level)
                mask = self.models [level] (paired_t)
                mask = mask.cpu ().numpy () [0][0] * 255
        return mask

    def paired_aug (self, raw, mask, lbl):
        paired = self.append_concat_last ([mask, lbl])
        aug = A.Compose([
            A.HorizontalFlip (),
            A.RandomRotate90 (p=0.5),
            A.VerticalFlip (),
            A.Transpose (),
            # A.ElasticTransform(p=0.75, alpha=100, sigma=240 * 0.05, alpha_affine=100 * 0.03),
            # A.OpticalDistortion(p=1, distort_limit=0.05, shift_limit=0.1),
        ], p=0.8)

        ret = aug (image=raw, mask=paired)
        raw, paired = ret ['image'], ret['mask']
        return (raw, paired [...,0], paired [...,1])

    def mask_aug (self, mask):
        aug = A.Compose ([
            A.Blur (p=0.5),
            A.Cutout(num_holes=3, max_h_size=mask.shape[0]//2, max_w_size=mask.shape[1]//2, p=0.1)
        ], p=1)
        return aug (image=mask, mask=None) ['image']

    def raw_aug (sefl, raw):
        aug = A.Compose ([
            A.RandomGamma(p=1, gamma_limit=(30, 236)),
            A.RandomContrast(p=0.8),
            A.GaussNoise (p=0.5),
            A.Blur (p=0.5)
        ], p=1)
        ret = aug (image=raw, mask=None) ['image']
        return ret


    def __getitem__ (self, idx):
        z0 = self.rng.randint (0, len (self.raw_list))
        raw_ret = []
        lbl_ret = []
        mask_ret = []

        raw_ret.append (copy.deepcopy (self.raw_list[0]))
        lbl_ret.append (copy.deepcopy (self.lbl_list[0]))

        mask_ret.append (self.predict (raw_ret [0], np.zeros_like (raw_ret[0]), 0))
        # mask_ret.append (np.zeros_like (raw_ret [0]))

        for level in range (self.max_level):
            H, W = self.sample_sizes [level]
            raw, lbl, mask = self.crop_from_prev (raw_ret, lbl_ret, mask_ret, H, W)

            if (self.rng.rand () < 0.33):
                mask = np.zeros_like (raw)
            
            p_new_mask = 0.5
            if (self.rng.rand () < p_new_mask or np.max (mask) == 0):
                mask = self.predict (raw, mask, level)
                # mask = np.zeros_like (raw)

            p_zero_mask = 0.15
            if (level == 0):
                p_zero_mask = 0.6
            if (self.rng.rand () < p_zero_mask):
                mask = np.zeros_like (raw)
            
            raw_ret += [raw]
            lbl_ret += [lbl]
            mask_ret += [mask]
            # print ("raw")
            # plt.imshow (raw, cmap='gray')
            # plt.show ()
            # print ("mask")
            # plt.imshow (mask, cmap='gray')
            # plt.show ()

        ret = {}

        for level in range (self.max_level):
            trippled = self.paired_aug (raw_ret [level+1], mask_ret [level+1], lbl_ret [level+1])
            raw_ret [level+1], mask_ret [level+1], lbl_ret [level+1] = trippled
            mask_ret [level + 1] = self.mask_aug (mask_ret [level + 1])
            raw_ret [level + 1] = self.raw_aug (raw_ret [level + 1].astype (np.uint8))
            paired = np.concatenate ([raw_ret[level+1][None], mask_ret[level+1][None]], 0)
            ret ['raw:' + str (level)] = paired
            ret ['lbl:' + str (level)] = lbl_ret [level + 1][None]

        return  ret   


