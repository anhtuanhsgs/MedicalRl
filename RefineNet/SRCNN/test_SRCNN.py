from SRCNN import *
import torch
import numpy as np
from operator import mul
from functools import reduce



net = SRCNN (2, 1)

x = np.ones ((3, 2, 255, 255), dtype=np.uint8)
x_t = torch.tensor (x, dtype=torch.float32)


out = net (x_t)
print (out.shape)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters)

# def check ():
# 	for i in range (10):
# 		yield i

# a = check ()

net = LightUnet (2, 1)
x = np.ones ((3, 2, 256, 256), dtype=np.uint8)
x_t = torch.tensor (x, dtype=torch.float32)
out = net (x_t)
print (out.shape)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters)