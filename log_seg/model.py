import torch
import torch.nn as nn
import numpy as np

class Dilated_Module (nn.Module):
    def __init__ (self, length, kernel_size, in_channels, out_channels):
        super (Dilated_Module, self).__init__ ()
        self.module_list = nn.ModuleList ()
        for i in range (length):
            nchannels = in_channels if i == 0 else out_channels
            self.module_list += [
                nn.Sequential (
                    nn.Conv3d (in_channels= nchannels, out_channels=out_channels, kernel_size=kernel_size, 
                        stride=1, dilation=2**i, padding=2**i, bias=False),
                    nn.ReLU ()
                )
            ]
        self.out_channels = out_channels

    def forward (self, x):
        outs = []
        for layer in self.module_list:
            if len (outs) == 0:
                outs.append (layer (x))
            else:
                outs.append (layer (outs[-1]))
        ret = outs[1] + outs[-1]

        return ret

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions, input_shape):
        super(DQN, self).__init__()
        self.dilate1 = Dilated_Module (length=4, kernel_size=3, in_channels=in_channels, out_channels=32) #Out: 96 ^ 3
        self.M1 = nn.MaxPool3d (kernel_size=4, stride=4, padding=1) # Out: 96 ^ 3
        self.dilate2 =  Dilated_Module (length=4, kernel_size=3, in_channels=32, out_channels=64) # Out: 24 ^ 3
        self.M2 = nn.MaxPool3d (kernel_size=3, stride=2, padding=1) # Out: 12 ^ 3
        self.dilate3 = Dilated_Module (length=4, kernel_size=3, in_channels=64, out_channels=64) # out = 12 ^ 3
        self.M3 = nn.MaxPool3d (kernel_size=3, stride=2, padding=1) # Out: 6 ^ 3
        dim = input_shape [0] // 4 // 2 // 2
        self.fc1 = nn.Linear(in_features=dim*dim*dim*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        self.insNorm1 = nn.InstanceNorm3d (32)
        self.insNorm2 = nn.InstanceNorm3d (64)
        self.insNorm3 = nn.InstanceNorm3d (64)


        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.insNorm1 (self.M1 (self.dilate1 (x)))
        x = self.insNorm2 (self.M2 (self.dilate2 (x)))
        x = self.insNorm3 (self.M3 (self.dilate3 (x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    from torch.autograd import Variable
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dqn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (dqn_device)
    net = DQN (5, 5).cuda ()
    tmp = torch.tensor (np.zeros ((2, 5, 96, 96, 96)), dtype=torch.float, device=dqn_device)
    # print (tmp)
    print (net (tmp).shape)





