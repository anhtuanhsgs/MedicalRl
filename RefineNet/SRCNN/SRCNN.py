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


class SRCNN (nn.Module):
    def __init__ (self, in_ch, out_ch):
        super (SRCNN, self).__init__ ()
        self.l1 = nn.Sequential (nn.Conv2d (in_ch, 64, 9, bias=True, padding=4), nn.ReLU ())
        self.l2 = nn.Sequential (nn.Conv2d (64, 32, 1, bias=True, padding=0), nn.ReLU ())
        self.l3 = nn.Sequential (nn.Conv2d (32, out_ch, 5, bias=True, padding=2), nn.ReLU ())
        self.last = nn.Sigmoid ()


    def forward (self, x):
        o1 = self.l1 (x)
        o2 = self.l2 (o1)
        o3 = self.l3 (o2)
        out = self.last (o3)
        return out



