from Unet import *
import torch
import numpy as np
from operator import mul
from functools import reduce

FEATURES = [8, 16, 32, 64]
net = Unet (2, FEATURES, 1)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters) 