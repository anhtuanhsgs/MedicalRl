from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init


class CNN(torch.nn.Module):
    def __init__(self, input_shape, num_action):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape [0], 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d (64, 128, 3, stride=1, padding=1)
        self.maxp5 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Conv2d (128, 256, 3, stride=1, padding=1)
        self.maxp6 = nn.MaxPool2d(2, 2)
        self.actor_conv = nn.Conv2d (256, num_action, 3, stride=1, padding=1)


        # self.lstm = nn.LSTMCell(1024, 512)
        # num_outputs = action_space.n
        num_values = input_shape [1] // (2 ** 6) * input_shape [2] // (2 ** 6) * 256
        self.policy_shape = (num_action, input_shape [1]//(2**6), input_shape [2]//(2**6))
        self.critic_linear = nn.Linear(num_values, 1)
        # self.actor_linear = nn.Linear(num_values, 2)

        # self.apply(weights_init)
        # relu_gain = nn.init.calculate_gain('relu')
        # self.conv1.weight.data.mul_(relu_gain)
        # self.conv2.weight.data.mul_(relu_gain)
        # self.conv3.weight.data.mul_(relu_gain)
        # self.conv4.weight.data.mul_(relu_gain)
        # self.conv5.weight.data.mul_(relu_gain)
        # self.conv6.weight.data.mul_(relu_gain)
        # # self.actor_linear.weight.data = norm_col_init(
        # #     self.actor_linear.weight.data, 0.01)
        # # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = norm_col_init(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)
        # self.actor_conv.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = F.relu(self.maxp5(self.conv5(x)))
        x = F.relu(self.maxp6(self.conv6(x)))
        critic = self.critic_linear(x.view(x.size(0), -1))
        actor = self.actor_conv (x).view (x.size (0), -1)
        return critic, actor

class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_outputs)

        # self.apply(weights_init)
        # relu_gain = nn.init.calculate_gain('relu')
        # self.conv1.weight.data.mul_(relu_gain)
        # self.conv2.weight.data.mul_(relu_gain)
        # self.conv3.weight.data.mul_(relu_gain)
        # self.conv4.weight.data.mul_(relu_gain)
        # self.actor_linear.weight.data = norm_col_init(
        #     self.actor_linear.weight.data, 0.01)
        # self.actor_linear.bias.data.fill_(0)
        # self.critic_linear.weight.data = norm_col_init(
        #     self.critic_linear.weight.data, 1.0)
        # self.critic_linear.bias.data.fill_(0)

        # self.lstm.bias_ih.data.fill_(0)
        # self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(inputs)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
