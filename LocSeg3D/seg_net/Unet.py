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




class Residual_Conv (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Residual_Conv, self).__init__ ()

		self.batch_norm_i = nn.InstanceNorm3d (in_ch) 
		self.conv1_i = nn.Sequential (nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1), nn.ReLU ())

		self.conv2_i = nn.Sequential (nn.Conv3d (in_ch, out_ch // 2, 3, bias=False, padding=1), nn.ReLU ())
		self.conv2_m = nn.Sequential (nn.Conv3d (out_ch // 2, out_ch, 3, bias=False, padding=1), nn.ReLU ())
		self.conv2_o = nn.Sequential (nn.ConvTranspose3d (out_ch, out_ch, 1, padding=0, bias=False), nn.ReLU ())

		self.conv1_o = nn.Sequential (nn.Conv3d (out_ch, out_ch, 3, bias=False, padding=1), nn.ReLU ())
		self.instance_norm_o = nn.InstanceNorm3d (out_ch)

	def forward (self, x):
		_in = self.batch_norm_i (x)
		x = self.conv1_i (_in)
		x = self.conv2_i (x)
		x = self.conv2_m (x)
		x = self.conv2_o (x)

		x = self.conv1_o (x)
		_out = self.instance_norm_o (x)
		return _in + _out


class Down (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Down, self).__init__ ()

		self.down_module = nn.Sequential (
			nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1),
			nn.ReLU (),
			Residual_Conv (out_ch, out_ch),
			nn.Conv3d (out_ch, out_ch, 3, bias=False, padding=1),
			nn.ReLU (),
			nn.InstanceNorm3d (out_ch)
		)

	def forward (self, x):
		x = self.down_module (x)
		return x

class Up (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Up, self).__init__ ()		

		self.up_module = nn.Sequential (
			nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1),
			nn.ReLU (),
			Residual_Conv (out_ch, out_ch),
			nn.ConvTranspose3d (out_ch, out_ch, 1, padding=0, bias=False),
			nn.ReLU (),
			nn.InstanceNorm3d (out_ch)
		)
	def forward (self, x):
		x = self.up_module (x)
		return x

class Unet (nn.Module):

	def __init__ (self, in_ch, features, out_ch):
		super (Unet, self).__init__ ()

		self.down1 = Down (in_ch, features[0])import math
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




class Residual_Conv (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Residual_Conv, self).__init__ ()

		self.batch_norm_i = nn.InstanceNorm3d (in_ch) 
		self.conv1_i = nn.Sequential (nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1), nn.ReLU ())

		self.conv2_i = nn.Sequential (nn.Conv3d (in_ch, out_ch // 2, 3, bias=False, padding=1), nn.ReLU ())
		self.conv2_m = nn.Sequential (nn.Conv3d (out_ch // 2, out_ch, 3, bias=False, padding=1), nn.ReLU ())
		self.conv2_o = nn.Sequential (nn.ConvTranspose3d (out_ch, out_ch, 1, padding=0, bias=False), nn.ReLU ())

		self.conv1_o = nn.Sequential (nn.Conv3d (out_ch, out_ch, 3, bias=False, padding=1), nn.ReLU ())
		self.batch_norm_o = nn.InstanceNorm3d (out_ch)

	def forward (self, x):
		_in = self.batch_norm_i (x)
		x = self.conv1_i (_in)
		x = self.conv2_i (x)
		x = self.conv2_m (x)
		x = self.conv2_o (x)

		x = self.conv1_o (x)
		_out = self.batch_norm_o (x)
		return _in + _out


class Down (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Down, self).__init__ ()

		self.down_module = nn.Sequential (
			nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1),
			nn.ReLU (),
			Residual_Conv (out_ch, out_ch),
			nn.Conv3d (out_ch, out_ch, 3, bias=False, padding=1, stride=2),
			nn.ReLU (),
			nn.InstanceNorm3d (out_ch)
		)

	def forward (self, x):
		x = self.down_module (x)
		return x

class Up (nn.Module):
	def __init__ (self, in_ch, out_ch):
		super (Up, self).__init__ ()		

		self.up_module = nn.Sequential (
			nn.Conv3d (in_ch, out_ch, 3, bias=False, padding=1),
			nn.ReLU (),
			Residual_Conv (out_ch, out_ch),
			nn.ConvTranspose3d (out_ch, out_ch, 2, stride=2, padding=0, bias=False),
			nn.ReLU (),
			nn.InstanceNorm3d (out_ch)
		)
	def forward (self, x):
		x = self.up_module (x)
		return x

class Unet (nn.Module):

	def __init__ (self, in_ch, features, out_ch):
		super (Unet, self).__init__ ()
		self.first_layer = nn.Sequential (
			nn.Conv3d (in_ch, features[0], 3, padding=1),
			nn.InstanceNorm3d (features[0])
		)
		self.down1 = Down (features[0], features[0])
		self.down2 = Down (features[0], features[1])
		self.down3 = Down (features[1], features[2])
		self.down4 = Down (features[2], features[3])
		self.middle = nn.Dropout (p=0.5, inplace=True)
		self.up4 = Up (features[3], features[2])
		self.up3 = Up (features[2], features[1])
		self.up2 = Up (features[1], features[0])
		self.up1 = Up (features[0], features[0])
		self.last_layer = nn.Sequential (
			nn.Conv3d (features[0], out_ch, 3, padding=1),
			nn.Sigmoid ()
		)

	def forward (self, x):
		x = self.first_layer (x)
		down1 = self.down1 (x)
		down2 = self.down2 (down1)
		down3 = self.down3 (down2)
		down4 = self.down4 (down3)
		middle = self.middle (down4)
		up4 = self.up4 (middle)
		up3 = self.up3 (up4 + down3)
		up2 = self.up2 (up3 + down2)
		up1 = self.up1 (up2 + down1)

		out = self.last_layer (up1 + x)
		return out
		self.down2 = Down (features[0], features[1])
		self.down3 = Down (features[1], features[2])
		self.down4 = Down (features[2], features[3])
		self.middle = nn.Dropout (p=0.5, inplace=True)
		self.up4 = Up (features[3], features[2])
		self.up3 = Up (features[2], features[1])
		self.up2 = Up (features[1], features[0])
		self.up1 = Up (features[0], features[0])
		self.last_layer = nn.Sequential (
			nn.Conv3d (features[0], out_ch, 3, padding=1),
			nn.Sigmoid ()
		)

	def forward (self, x):
		down1 = self.down1 (x)
		down2 = self.down2 (down1)
		down3 = self.down3 (down2)
		down4 = self.down4 (down3)
		middle = self.middle (down4)
		up4 = self.up4 (middle)
		up3 = self.up3 (up4 + down3)
		up2 = self.up2 (up3 + down2)
		up1 = self.up1 (up2 + down1)

		out = self.last_layer (up1 + x)
		return out