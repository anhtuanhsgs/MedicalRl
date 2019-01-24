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

class ConvModule2D (nn.Module):
    def __init__ (self, length, kernel_size, in_channels, out_channels):
        super (ConvModule2D, self).__init__ ()
        self.batch_norm_i = nn.BatchNorm2d (in_channels) 
        self.module_list = nn.ModuleList ()
        for i in range (length):
            nchannels = in_channels if i == 0 else out_channels
            self.module_list += [
                nn.Sequential (
                    nn.Conv2d (in_channels= nchannels, out_channels=out_channels, kernel_size=kernel_size, 
                        stride=1, padding=1, bias=False),
                    nn.ReLU ()
                )
            ]
        self.batch_norm_o = nn.BatchNorm2d (out_channels)
        self.out_channels = out_channels

    def forward (self, x):
        outs = []
        x = self.batch_norm_i (x)
        for layer in self.module_list:
            if len (outs) == 0:
                outs.append (layer (x))
            else:
                outs.append (layer (outs[-1]))
        ret = outs[1] + outs[-1]
        ret = self.batch_norm_o (ret)
        return ret   

class DilatedConvModule2D (nn.Module):
    def __init__ (self, length, kernel_size, in_channels, out_channels):
        super (DilatedConvModule2D, self).__init__ ()
        self.batch_norm_i = nn.BatchNorm2d (in_channels) 
        self.module_list = nn.ModuleList ()
        for i in range (length):
            nchannels = in_channels if i == 0 else out_channels
            self.module_list += [
                nn.Sequential (
                    nn.Conv2d (in_channels= nchannels, out_channels=out_channels, kernel_size=kernel_size, 
                        stride=1, dilation=2**i, padding=2**i, bias=False),
                    nn.ReLU ()
                )
            ]
        self.batch_norm_o = nn.BatchNorm2d (out_channels)
        self.out_channels = out_channels

    def forward (self, x):
        outs = []
        x = self.batch_norm_i (x)
        for layer in self.module_list:
            if len (outs) == 0:
                outs.append (layer (x))
            else:
                outs.append (layer (outs[-1]))
        ret = outs[1] + outs[-1]
        ret = self.batch_norm_o (ret)
        return ret



class FCN2D (nn.Module):
    def __init__ (self, in_ch, out_ch):
        super (FCN2D, self).__init__ ()
        self.module1 = ConvModule2D (length=4, kernel_size=3, in_channels=in_ch, out_channels=32)
        self.module2 = DilatedConvModule2D (length=4, kernel_size=3, in_channels=32, out_channels=64)
        self.module3 = DilatedConvModule2D (length=4, kernel_size=3, in_channels=64, out_channels=32)
        self.final = nn.Sequential (
            nn.Conv2d (in_channels=32, out_channels=out_ch, kernel_size=3, padding=1),
            nn.Sigmoid ()
        )


    def forward (self, x):
        out1 = self.module1 (x)
        out2 = self.module2 (out1)
        out3 = self.module3 (out2)
        out = self.final (out1 + out3)
        return out