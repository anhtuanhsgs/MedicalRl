from Unet import *
import torch
import numpy as np
from operator import mul
from functools import reduce

FEATURES = [8, 16, 32, 64]
net = Unet (2, FEATURES, 1)

x = np.ones ((1, 2, 64, 64, 64), dtype=np.uint8)
x_t = torch.tensor (x, dtype=torch.float32)
out = net (x_t)
print (out.shape)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters) 