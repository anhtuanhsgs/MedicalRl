import numpy as np 
from skimage.measure import label
from skimage.morphology import skeletonize
from scipy.ndimage.morphology import binary_erosion
import random
from collections import namedtuple
import skimage.io as io
import os, sys, argparse, glob
from natsort import natsorted 
from img_aug_func import *
import copy
import math
from seg_net.Unet import *
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym

import gym.spaces

PAD = 80
MAX_LV = 3
CHANNELS = 5
SEG_NET_SIZE = (64, 64, 64, 1)
RL_NET_SIZE = (96, 96, 96, CHANNELS)
ORIGIN_SIZE = (128, 128, 128)
TREE_BASE = 9
NUM_ACTIONS = 9 + 2 # 9 directions, 1 back, 1 predict

class Action:
    def __init__ (self, value):
        self.val = value
        self._len = 11

    def __len__ (self):
        return 11

    def numpy (self):
        ret = np.zeros (11, dtype=np.float32)
        ret [self.val] = 1
        return ret

class ActionSpace (gym.spaces.Discrete):
    def __init__ (self, length):
        self._len = length
        self.n = length

    def numpy (self, val):
        ret = np.zeros (self._len, dtype=np.float32)
        ret [self.val] = 1
        return ret

    def sample (self):
        return random.randint (0, self._len - 1) 

    def num_actions (self):
        return self.n

class ObservationSpace (gym.spaces.Box):
    def __init__ (self):
        # Mask + EM
        self.shape = RL_NET_SIZE

class Node:
    def __init__ (self, size = list (ORIGIN_SIZE), start=[80, 80, 80], size_decay=0.7):
        self.id = 1
        self.start = start
        self.size = size
        self.level = 0
        self.size_decay = size_decay
        self.dz = []; self.dy = []; self.dx = [];
        for _z in range (2):
            for _y in range (2):
                for _x in range (2):
                    self.dz += [1.0 * _z]
                    self.dy += [1.0 * _y]
                    self.dx += [1.0 * _x]
        self.dz += [0.5];
        self.dy += [0.5];
        self.dx += [0.5];
        
        # dz = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5]
        # dy = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5]
        # dx = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]

        self.tree_base = len (self.dx);
        # print (self.tree_base)

        self.his = []

    def step (self, action):
        start, size = self.start, self.size
        # print ('start, size', start, size)
        # print ('id', self.id, 'depth', self.level, 'action', action)
        if action < self.tree_base:
            #Force choose predict if reach lv 7
            if self.level == MAX_LV:
                self.level -= 1
                self.start, self.size = self.his.pop ()
                self.id = self.id // self.tree_base
                return
            self.id = self.id * self.tree_base + action
            self.his += [copy.deepcopy ((start, size))]

            #Update window position and size
            start [0] += int (self.dy [action] * (1 - self.size_decay) * self.size [0])
            start [1] += int (self.dx [action] * (1 - self.size_decay) * self.size [1])
            start [2] += int (self.dz [action] * (1 - self.size_decay) * self.size [2])

            size [0] = int (size[0] * self.size_decay)
            size [1] = int (size[1] * self.size_decay)
            size [2] = int (size[2] * self.size_decay)

            self.start, self.size = start, size
            self.level += 1
        else:
            if self.level == 0:
                return
            self.id = self.id // self.tree_base
            self.start, self.size = self.his.pop ()
            self.level -= 1

    def center (self):
        return ((self.start[0] + self.size [0]) // 2, 
            (self.start[1] + self.size [1]) // 2, 
            (self.start[2] + self.size [2]) // 2)

class State:
    def __init__ (self, node):
        self.node = node
        self.done = False
        self.action_his = []

    def debug (self):
        print ('id:', self.id, 'start:', self.start, 'size:', self.size,
            'target:', self.target, 'depth:', self.depth)

class Environment:
    def __init__ (self, raw_list, lbl_list):
        self.action_space = ActionSpace (NUM_ACTIONS)
        self.observation_space = ObservationSpace ()
        self.obs_shape = RL_NET_SIZE
        self.mask_shape = RL_NET_SIZE [:3]
        self.raw_list = raw_list
        self.lbl_list = lbl_list
        self.viewed = {}
        self.tree_base = TREE_BASE

        self.base_start = (80, 80, 80)
        self.base_size = ORIGIN_SIZE

        self.device = torch.device("cuda:1")
        self.setup_net ()

    def setup_net (self):
        self.net = Unet (1, [8, 16, 32, 64], 1).to (self.device)
        checkpoint_path = 'seg_net/checkpoints/checkpoint_364500.pth.tar'
        checkpoint = torch.load  (checkpoint_path)
        self.net.load_state_dict (checkpoint['state_dict'])

    def calculate_score (self, gt_lbl, mask, cell_id):
        gt_mask = (gt_lbl == cell_id).astype (np.int32)
        mask = (mask > 128).astype (np.int32)

        #plt.imshow (gt_mask, cmap='gray')
        #plt.show ()
        #plt.imshow (mask, cmap='gray')
        #plt.show ()
        # print (gt_mask.shape, mask.shape)

        dice_score = np.sum(mask[gt_mask==1])*2.0 / (np.sum(mask) + np.sum(gt_mask))
        return dice_score

    def get_cell_id (self, gt_lbl):
        ids, count = np.unique (gt_lbl, return_counts=True)
        count [0] = -1
        biggest_cell_id = ids [np.argmax (count)]
        return biggest_cell_id

    def predict (self):
        # print ("predict")
        st = self.state.node.start
        sz = self.state.node.size
        padding = 30
        with torch.no_grad():
            raw_patch = self.raw_list [st[0]: st[0]+sz[0], 
                                    st[1]: st[1] + sz[1], 
                                    st[2]: st[2] + sz[2]]
            raw_patch = resize (raw_patch, SEG_NET_SIZE [:3], order=0, mode='reflect', preserve_range=True)
            raw_patch = np.expand_dims (np.expand_dims (raw_patch, 0), 0).astype (np.float32) / 255.0
            new_mask = self.net (torch.tensor (raw_patch).to (self.device))
            new_mask = np.squeeze (new_mask.cpu ().numpy()) * 255

        new_mask = resize (new_mask, sz, order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        cur_mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1], 
                        st[2]: st[2] + sz[2]]
        self.mask [st[0]: st[0]+sz[0], 
                    st[1]: st[1] + sz[1], 
                    st[2]: st[2] + sz[2]] = np.maximum (cur_mask, new_mask)
        gt_lbl = self.lbl_list [st[0]-padding: st[0]+sz[0]+padding, 
                            st[1]-padding: st[1] + sz[1]+padding, 
                            st[2]-padding: st[2] + sz[2]+padding]

        new_mask = np.pad (new_mask, padding, mode='constant', constant_values=0)
        cell_id = self.get_cell_id (self.lbl_list [st[0]: st[0]+sz[0], 
                                        st[1]: st[1] + sz[1], 
                                        st[2]: st[2] + sz[2]])
        # Hanle revisit segmented cell
        if cell_id in self.viewed:
            return 0
        self.viewed [cell_id] = True
        score = self.calculate_score (gt_lbl, new_mask, cell_id)
        if score < 0.6:
            score = 0
        return score

    def update_taboo (self):
        self.taboo [self.state.node.id] = True
        
    def step (self, action):
        done = False
        reward = 0
        state = copy.deepcopy (self.state)
        self.set_state (state)
        prev_state = copy.deepcopy (state)

        action = Action (action)
        info = { 'ale.lives': 1}
        
        # Handle revisiting a visited node
        taboo_action = False
        if (0 <= action.val < self.tree_base):
            if (self.state.node.id * TREE_BASE + action.val) in self.taboo:
                taboo_action = True
                action.val = self.tree_base + 1

        if action.val == self.tree_base:
            reward = self.predict ()
        if action.val >= self.tree_base:
            self.update_taboo ()
        if (action.val < self.tree_base and state.node.level == MAX_LV) or taboo_action:
            self.update_taboo ()


        if (action.val >= self.tree_base and state.node.level == 0):
            done = True
            info = { 'ale.lives': 0}
            return self.observation (), reward, done, info

        state.node.step (action.val)

        self.set_state (state)
        ret = self.observation (), reward, done, info
        return ret

    def sample_action (self):
        return self.action_space.sample_action ()       

    def reset (self):
        self.state = State (Node ())
        self.mask = np.zeros (self.mask_shape, dtype=np.uint8)
        self.mask = np.pad (self.mask, 80, mode='constant', constant_values=0)
        self.taboo = {}
        self.viewed = {}
        return self.observation ()


    def set_state (self, state):
        self.state = copy.deepcopy (state)

    def location_mask (self):
        location = np.zeros (ORIGIN_SIZE, dtype=np.uint8)
        st = self.state.node.start
        sz = self.state.node.size

        location [st[0]-80: st[0]-80+sz[0], 
                st[1]-80: st[1]-80+sz[1], 
                st[2]-80: st[2]-80+sz[2]] = 255
        return location

    def observation (self):
        '''
            Observation of size (D, H, W, 5)
        '''
        st = self.state.node.start
        sz = self.state.node.size
        raw_patch = self.raw_list [st[0]: st[0] + sz[0], 
                                    st[1]: st[1] + sz[1], 
                                    st[2]: st[2] + sz[2]]
        mask_patch = self.mask [st[0]: st[0] + sz[0], 
                                st[1]: st[1] + sz[1], 
                                st[2]: st[2] + sz[2]]
        raw_patch = resize (raw_patch, RL_NET_SIZE [:3], order=0, mode='wrap', preserve_range=True)
        mask_patch = resize (mask_patch, RL_NET_SIZE [:3], order=0, mode='wrap', preserve_range=True)
        raw_patch = np.expand_dims (raw_patch, -1)
        mask_patch = np.expand_dims (mask_patch, -1)

        # print ('base', self.base_start, self.base_size, '\\base')

        full_raw = self.raw_list [self.base_start [0]: self.base_start [0] + self.base_size[0],
                            self.base_start [1]: self.base_start [1] + self.base_size[1],
                            self.base_start [2]: self.base_start [2] + self.base_size[2]]

        full_raw = resize (full_raw, RL_NET_SIZE [:3], order=0, mode='wrap', preserve_range=True)

        full_raw = np.expand_dims (full_raw, -1)

        full_mask = self.mask [self.base_start [0]: self.base_start [0] + self.base_size[0],
                            self.base_start [1]: self.base_start [1] + self.base_size[1],
                            self.base_start [2]: self.base_start [2] + self.base_size[2]]
        full_mask = resize (full_mask, RL_NET_SIZE [:3], order=0, mode='wrap', preserve_range=True)

        full_mask = np.expand_dims (full_mask, -1)

        location_mask = self.location_mask ();
        location_mask = resize (location_mask, RL_NET_SIZE [:3], order=0, mode='wrap', preserve_range=True)
        location_mask = np.expand_dims (location_mask, -1)

        ret = np.concatenate ([full_raw, full_mask, location_mask, raw_patch, mask_patch], -1)
        return ret;

    def concat_last_dim_2_x (self, arr):
        # Arr of HxWxC
        # Ret of HxW*C
        assert (len (arr.shape) == 3)
        ret = []
        for i in range (arr.shape [-1]):
            ret.append (arr[...,i])
        ret = np.concatenate (ret, -1)
        return ret

    def render (self):
        ret = self.observation ();

        st = self.state.node.start
        sz = self.state.node.size

        z = int (1.0 * (st[0] + sz[0] // 2 - 80) / ORIGIN_SIZE [0] * SEG_NET_SIZE[0])
        y = int (1.0 * (st[1] + sz[1] // 2 - 80) / ORIGIN_SIZE [0] * SEG_NET_SIZE[0])
        x = int (1.0 * (st[2] + sz[2] // 2 - 80) / ORIGIN_SIZE [0] * SEG_NET_SIZE[0])
        yx = self.concat_last_dim_2_x (ret [z, :, :, :])
        zx = self.concat_last_dim_2_x (ret [:, y, :, :])
        zy = self.concat_last_dim_2_x (ret [:, :, x, :])
        # Concatenate to Y dim
        ret = np.concatenate ([yx, zx, zy], 0)
        ret = np.repeat (np.expand_dims (ret, -1), 3, axis=-1)

        return ret.astype (np.uint8)


