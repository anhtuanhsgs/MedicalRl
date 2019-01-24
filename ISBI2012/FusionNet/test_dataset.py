import os, sys, argparse, glob
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import resize
from skimage.morphology import erosion, binary_erosion
from skimage.filters import sobel

from ISBI2012_Dataset import *
from FusionNet import FusionNet
from img_aug_func import *
from Logger import Logger
import multiprocessing as mp
from tqdm import tqdm
import time

import os

import matplotlib.pyplot as plt

models_paths = [
	'checkpoints/0_1.0/checkpoint_226500.pth.tar',
	'checkpoints/1_1.0/checkpoint_451500.pth.tar',
	'checkpoints/2_1.0/checkpoint_706500.pth.tar',
	'checkpoints/3_1.0/checkpoint_759000.pth.tar'
]

FEATURES = [16, 32, 64, 128]
IN_CH = 2
OUT_CH = 1
ZOOM_FACTOR = 0.6

cuda = torch.device ("cuda")
cuda0 = torch.device('cuda:0')

def get_data (path, downsample=1):
    train_path = natsorted (glob.glob(path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)
    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train

models = []
for level in range (4):
	model = FusionNet (IN_CH, FEATURES, OUT_CH).to (cuda0)
	checkpoint = torch.load  (models_paths [level])
	model.load_state_dict (checkpoint['state_dict'])
	models.append (model)

X_train, y_train = get_data (path='../DATA/train/')
isbi2012 = ISBI2012_Dataset16_refine ('test', X_train, y_train, 4, ZOOM_FACTOR, models, [0, 0, 0, 0])
dataset = DataLoader (isbi2012, batch_size=1, num_workers=)
for sample in dataset:
	for level in range (4):
		print ("level:", level)
		raw = sample ['raw:' + str (level)]
		lbl = sample ['lbl:' + str (level)]
		print (raw.shape, lbl.shape)
		raw, mask = raw[0][0], raw[0][1]
		lbl = lbl [0]
		img = np.concatenate ([raw, mask, lbl], 1)
		plt.imshow (img, cmap='gray')
		plt.show ()


