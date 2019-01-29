from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import norm_col_init, weights_init

class A3Clstm(torch.nn.Module):
    def __init__(self, input_shape, num_action, hidden_feat=256):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(input_shape [0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.norm1 = nn.InstanceNorm2d (32)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.norm2 = nn.InstanceNorm2d (32)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=2)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.norm3 = nn.InstanceNorm2d (64)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.norm4 = nn.InstanceNorm2d (64)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.maxp5 = nn.MaxPool2d(2, 2)
        self.norm5 = nn.InstanceNorm2d (128)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.maxp6 = nn.MaxPool2d(2, 2)
        self.norm6 = nn.InstanceNorm2d (256)

        num_values = input_shape[1] // (2 ** 6) * input_shape[2] // (2 ** 6) * 128
        self.lstm = nn.LSTMCell(num_values, hidden_feat)
        self.critic_linear = nn.Linear(hidden_feat, 1)
        self.actor_linear = nn.Linear(hidden_feat, num_action)


        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.norm1 (self.maxp1(self.conv1(inputs))))
        x = F.relu(self.norm2 (self.maxp2(self.conv2(x))))
        x = F.relu(self.norm3 (self.maxp3(self.conv3(x))))
        x = F.relu(self.norm4 (self.maxp4(self.conv4(x))))
        x = F.relu(self.norm5 (self.maxp5(self.conv5(x))))
        x = F.relu(self.norm6 (self.maxp6(self.conv6(x))))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


# def make_layers(cfg, instance_norm=True, in_channels=5):
#     layers = []
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if instance_norm:
#                 layers += [conv2d, nn.InstanceNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)

# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# class VGGlstm(nn.Module):

#     def __init__(self, input_shape, num_action):
#         super(VGGlstm, self).__init__()
#         self.features = features
        
#         num_values = input_shape [1] // (2 ** 5) * input_shape [2] // (2 ** 5) * 512

#         self.dense = nn.Sequential(
#             nn.Linear(num_values, 512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512, 1024),
#             nn.ReLU(True),
#             nn.Dropout(),
#         )

#         num_outputs = input_shape [1] * input_shape[2] * num_action_per_pixel
        
#         self.lstm = nn.LSTMCell (1024, 512)
#         self.critic_linear = nn.Linear (512, 1)
#         self.actor_linear = nn.Linear (512, num_outputs)

#     def forward(self, inputs):
#         x, (hx, cx) = inputs
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.dense (x)
#         hx, cx = self.lstm (x, (hx, cx))
#         x = hx
#         return self.critic_linear(x), self.actor_linear(x), (hx, cx)

def vgg16lstm (**kwargs):
    model = VGGlstm(make_layers(cfg['D']), **kwargs)
    return model

class SimpleCNN (nn.Module):
    def __init__(self, input_shape, num_action):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape [0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        num_values = input_shape[1] // (2 ** 4) * input_shape[2] // (2 ** 4) * 64
        print (num_values)
        self.dense1 = nn.Linear (num_values, 512)
        self.dense2 = nn.Linear (512, 1024)
        self.dense3 = nn.Linear (1024, 512)
        num_outputs = num_action
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

    def forward(self, inputs):
        x = inputs / 255.0
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)
        print (x.shape)

        x = self.dense1 (x)
        x = self.dense2 (x)
        x = self.dense3 (x)

        return self.critic_linear(x), self.actor_linear(x)

if __name__ == '__main__':
    model = SimpleCNN ((1, 256, 256), 8*8*2)
    a = np.ones ((1, 1, 256, 256))
    a_t = torch.tensor (a, dtype=torch.float32)
    b, c = model (a_t)
    print (b.shape, c.shape)
    