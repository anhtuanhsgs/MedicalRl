from FCN2D import *
import torch
import numpy as np

net = FCN2D (1, 1)

x = np.ones ((3, 1, 255, 255), dtype=np.uint8)
x_t = torch.tensor (x, dtype=torch.float32)


out = net (x_t)
print (out.shape)
