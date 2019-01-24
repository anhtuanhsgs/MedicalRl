import math, time, random
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse, glob, copy
sys.path.append('../')
from torch.utils.data import Dataset
from skimage.measure import label
from skimage.transform import resize

import cv2
import albumentations as A

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
		return self.iter_per_epoch

	def aug (self, image, mask):
		aug = Compose([
		            OneOf([ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_NEAREST, border_mode=4, always_apply=False, p=0.5),
		                   RandomSizedCrop(min_max_height=(400, 512), height=self.DIMY, width=self.DIMX, interpolation=cv2.INTER_NEAREST, p=0.5),
		                   PadIfNeeded(min_height=self.DIMY, min_width=self.DIMX, p=0.5)], p=1),    
		            VerticalFlip(p=0.5),              
		            RandomRotate90(p=0.5),
		            OneOf([
		                ElasticTransform(p=0.5, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
		                GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST),
		                OpticalDistortion(p=0.5, distort_limit=(0.05, 0.05), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST)                  
		                ], p=0.8),
		            CLAHE(p=0.8),
		            RandomContrast(p=0.8),
		            RandomBrightness(p=0.8),
		            RandomGamma(p=0.8)]
				)
		ret = aug (image=image, mask=mask)
		return ret ['image'], ret ['mask']

	def crop (self, image, mask):
		cropper = A.RandomCrop (height=size [1], width=size[2], p=1)
		ret = cropper (image=image, mask=mask)
		return ret ['image'], ret ['mask']

	def __getitem__ (self, idx):
		size = self.output_size
		z0 = self.rng.randint (0, len (self.raw_list))
		raw, lbl = copy.deepcopy (self.raw_list [z0], self.lbl_list [z0])
		raw, lbl = self.crop ()
		raw, lbl = self.aug (raw, lbl)
		self.mask = np.zeros_like (self.patch)
		raw = np.concatenate ([raw[None], mask[None]], 0)
		return {'raw': raw, 'lbl': lbl}



