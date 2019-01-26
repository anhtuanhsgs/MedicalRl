from model import *
import torch
import numpy as np
from operator import mul
from torch.autograd import Variable
import time
from functools import reduce 
# x = np.ones ((1, 1, 64, 64), dtype=np.uint8)
# x_t = torch.tensor (x, dtype=torch.float32)
# out = net (x_t)
# print (out.shape)

net = A3Clstm (input_shape=(5, 512, 512), num_action_per_pixel=2, hidden_feat=256)

num_parameters = 0
for parameter in net.parameters():
    op_shape = list (parameter.shape)
    num_parameters += reduce (mul, op_shape, 1)

print (num_parameters)
with torch.cuda.device (0):
    net = net.cuda ()
    x = torch.rand ((1, 5, 512, 512), dtype=torch.float32).cuda ()
    hx = Variable (torch.zeros (1, 256)).cuda ()
    cx = Variable (torch.zeros (1, 256)).cuda ()
    critic, actor, (hx, cx) = net ((x, (hx, cx)))
    print (critic.shape, actor.shape, hx.shape, cx.shape)


# CUDA run test
# with torch.cuda.device (0):
#     net = net.cuda ()

# while True:
#     with torch.cuda.device (0):
#         x = torch.rand ((1, 5, 1024, 1024), dtype=torch.float32).cuda ()
#         hx = Variable (torch.zeros (1, 512)).cuda ()
#         cx = Variable (torch.zeros (1, 512)).cuda ()
#         value, logit, (hx, cx) = net (inputs=(x, (hx, cx)))
#         print (value.shape, logit.shape, hx.shape, cx.shape)
#     time.sleep (1)


# CPU run test
# while True:
#     with torch.cuda.device (0):
#         x = torch.rand ((1, 5, 1024, 1024), dtype=torch.float32)
#         hx = Variable (torch.zeros (1, 512))
#         cx = Variable (torch.zeros (1, 512))
#         value, logit, (hx, cx) = net (inputs=(x, (hx, cx)))
#         print (value.shape, logit.shape, hx.shape, cx.shape)
#         tmp = torch.mean (torch.sum (value))
#     time.sleep (1)
