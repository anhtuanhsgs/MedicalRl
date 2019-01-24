import math, time, random
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse, glob, copy
from os import sys, path
from torch.utils.data import Dataset
from skimage.measure import label
from skimage.transform import resize
import cv2
import albumentations as A
sys.path.append('../')
from Utils.utils import *

import matplotlib.pyplot as plt


class EMDataset (Dataset):
    def __init__ (self, set_type, X, y, output_size):
        self.raw_list = X
        self.lbl_list = y
        self.n = len (X)
        self.set_type = set_type
        self.iter_per_epoch = self.n
        self.rng = np.random.RandomState (time_seed ())
        self.output_size = output_size

    def __len__ (self):
        return self.iter_per_epoch * 240

    def aug (self, image, mask):
        aug = A.Compose([
                    A.HorizontalFlip (),
                    A.VerticalFlip(p=0.5),              
                    A.RandomRotate90(p=0.5),
                    A.Transpose (),
                    A.OneOf([
                        A.ElasticTransform(p=0.5, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                        A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST),
                        # A.OpticalDistortion(p=0.5, distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST)                  
                        ], p=0.8),
                    A.RandomGamma(p=1, gamma_limit=(30, 236)),
                    A.RandomContrast(p=0.8),
                    A.GaussNoise (p=0.5),
                    A.Blur (p=0.5)]
                )
        ret = aug (image=image, mask=mask)
        return ret ['image'], ret ['mask']

    def crop (self, image, mask):
        size = self.output_size
        cropper = A.RandomCrop (height=size [0], width=size[1], p=1)
        ret = cropper (image=image, mask=mask)
        return ret ['image'], ret ['mask']

    def __getitem__ (self, idx):
        size = self.output_size
        z0 = self.rng.randint (0, len (self.raw_list))
        raw = copy.deepcopy (self.raw_list [z0])
        lbl = copy.deepcopy (self.lbl_list [z0])
        raw, lbl = self.crop (raw, lbl)
        raw, lbl = self.aug (raw, lbl)
        mask = np.zeros_like (raw)
        raw = np.concatenate ([raw[None], mask[None]], 0)
        lbl = np.expand_dims (lbl, 0)
        return {'raw': raw, 'lbl': lbl}



