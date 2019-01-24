import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np

def weighted_binary_cross_entropy(output, target, weights=None):
    ESP = 1e-5
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output + ESP)) + \
               weights[0] * ((1 - target) * torch.log(1 - output + ESP))
    else:
        loss = target * torch.log(output + ESP) + (1 - target) * torch.log(1 - output + ESP)

    return torch.neg(torch.mean(loss))