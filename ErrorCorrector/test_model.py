from model import CNN
import torch
import numpy as np
from operator import mul
from functools import reduce

# x = np.ones ((1, 1, 64, 64), dtype=np.uint8)
# x_t = torch.tensor (x, dtype=torch.float32)
# out = net (x_t)
# print (out.shape)

net = CNN ((3, 1024, 1024), 2)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters) 