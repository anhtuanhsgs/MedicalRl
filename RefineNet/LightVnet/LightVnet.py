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


class Up (nn.Module):
    def __init__ (self):
        super (Up, self).__init__ ()
        self.sub_pix = nn.PixelShuffle (2)
        
    def forward (self, x):
        x = self.sub_pix (x)
        return x


class Down (nn.Module):
    def __init__ (self, in_ch, out_ch):
        super (Down, self).__init__ ()
        self.pool = nn.MaxPool2d (2)
        self.conv = nn.Sequential (nn.Conv2d (in_ch, out_ch, 3, bias=True, padding=1), nn.ReLU ())
        self.norm = nn.InstanceNorm2d (out_ch)

    def forward (self, x):
        x = self.pool (x)
        x = self.conv (x)
        x = self.norm (x)
        return x

class LightVnet (nn.Module):
    def __init__ (self, in_ch, out_ch):
        super (LightVnet, self).__init__ ()
        self.first = nn.Sequential (nn.InstanceNorm2d (in_ch), nn.Conv2d (in_ch, 8, 3, bias=True, padding=1), nn.ReLU ())
        self.down1 = Down (8, 16)
        self.down2 = Down (16, 32)
        self.down3 = Down (32, 64)
        self.up3 = Up ()
        self.up2 = Up ()
        self.up1 = Up ()
        self.last = nn.Sequential (nn.Conv2d (64 // (4 ** 3), out_ch, 3, bias=True, padding=1), nn.Sigmoid ())

    def forward (self, x):
        e0 = self.first (x)
        e1 = self.down1 (e0)
        e2 = self.down2 (e1)
        e3 = self.down3 (e2)
        d2 = self.up3 (e3)
        d1 = self.up2 (d2)
        d0 = self.up1 (d1)
        out = self.last (d0)
        return out