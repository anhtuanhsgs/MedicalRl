import numpy as np 
from skimage.measure import label
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import adjusted_rand_score

import skimage.io as io
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

MAX_LV = 4 # 0, 1, 2
RL_CHANNELS = 4 # raw, mask, refine?, zoomed_patch?
SEG_CHANNELS = 1
RL_NET_SIZE = (64, 64, 64, RL_CHANNELS)
ORIGIN_SIZE = (64, 64, 64)

DZ = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5]
DY = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5]
DX = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]
TREE_BASE = len (DZ)
NUM_ACTIONS = len (DZ) + 2 # 5 zoom patches, refine?, back 
FEATURES = [8, 16, 32, 64]
SIZE_DECAY = 0.6
CELL_THRESHOLD = 200
SEG_NET_SIZE = (64, 64, 64)
UPDATE_MASK_PADDING = 8
BORDER_SIZE = 80

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
    def __init__ (self, shape):
        self.shape = shape

class Node:
    def __init__ (self, start):
        self.id = 1
        self.origin_start = copy.deepcopy (start)
        self.start = start
        self.size = list (ORIGIN_SIZE)
        self.level = 0
        SIZE_DECAY
        self.his = []

    def step (self, action):
        start, size = self.start, self.size
        # Trying to visit child
        if action < TREE_BASE:

            self.id = self.id * TREE_BASE + action
            self.his += [copy.deepcopy ((start, size))]

            #Update window position and size
            start [0] += int (DZ [action] * (1 - SIZE_DECAY) * self.size [0])
            start [1] += int (DY [action] * (1 - SIZE_DECAY) * self.size [1])
            start [2] += int (DX [action] * (1 - SIZE_DECAY) * self.size [2])

            size [0] = int (size[0] * SIZE_DECAY)
            size [1] = int (size[1] * SIZE_DECAY)
            size [2] = int (size[2] * SIZE_DECAY)

            self.start, self.size = start, size
            self.level += 1
        else:
            if self.level == 0:
                return
            self.id = self.id // TREE_BASE
            self.start, self.size = self.his.pop ()
            self.level -= 1

class State:
    def __init__ (self, node):
        self.node = node
        self.action_his = []

def jaccard_score (gt_lbl, mask):
    lbled_mask = label (mask > CELL_THRESHOLD, background=0)
    return jaccard_similarity_score (lbled_mask.flatten (), gt_lbl.flatten ())

def rand_score (gt_lbl, mask):
    lbled_mask = label (mask > CELL_THRESHOLD, background=0)
    return adjusted_rand_score (lbled_mask.flatten (), gt_lbl.flatten ())

def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class Environment:
    def __init__ (self, raw, lbl, SEG_checkpoints_path):
        self.action_space = ActionSpace (NUM_ACTIONS)
        self.observation_space = ObservationSpace (RL_NET_SIZE)
        self.obs_shape = RL_NET_SIZE
        self.raw = raw
        self.lbl = lbl
        self.raw = np.pad (self.raw, BORDER_SIZE, mode='constant', constant_values=0)
        self.lbl = np.pad (self.lbl, BORDER_SIZE, mode='constant', constant_values=0)

        self.base_start = (0, 0, 0)
        self.base_size = ORIGIN_SIZE
        self.thres = 128
        self.metric = rand_score
        self.vol_size = self.raw.shape
        
        self.rng = np.random.RandomState(time_seed ())
        self.device = torch.device("cuda:1")
        self.setup_nets (SEG_checkpoints_path)

    def setup_nets (self, path):
        self.seg_net_size = SEG_NET_SIZE
        self.net = Unet (SEG_CHANNELS, FEATURES, 1).to (self.device)
        checkpoint = torch.load  (path)
        self.net.load_state_dict (checkpoint['state_dict'])
        self.net.eval ()  


    def reset (self):
        z0 = self.rng.randint (BORDER_SIZE, self.vol_size [0] - ORIGIN_SIZE [0] - BORDER_SIZE)
        y0 = self.rng.randint (50+BORDER_SIZE, self.vol_size [1] - ORIGIN_SIZE [1] - BORDER_SIZE-50)
        x0 = self.rng.randint (180+BORDER_SIZE, self.vol_size [2] - ORIGIN_SIZE [2] - BORDER_SIZE-180)
        # print ((z0, y0, x0))
        # z0, y0, x0 = (114, 312, 290)
        #(114, 312, 290)
        self.state = State (Node (start=[z0, y0, x0]))
        self.mask = np.zeros (ORIGIN_SIZE, dtype=np.uint8)
        self.history = {}
        self.viewed = {}
        self.history [self.state.node.id] = {'refined':False, 'zoomed':0}
        self.stack = []
        return self.observation ()

    def get_cell_id (self, gt_lbl):
        ids, count = np.unique (gt_lbl, return_counts=True)
        count [0] = -1
        biggest_cell_id = ids [np.argmax (count)]
        return biggest_cell_id

    def calculate_score (self, gt_lbl, mask, cell_id):
        gt_mask = (gt_lbl == cell_id).astype (np.int32)
        mask = (mask > 128).astype (np.int32)
        dice_score = np.sum(mask[gt_mask==1])*2.0 / (np.sum(mask) + np.sum(gt_mask))
        return dice_score

    def refine (self):
        # print ('-----------------------Refine')
        st = self.state.node.start
        sz = self.state.node.size
        padding = UPDATE_MASK_PADDING
        net = self.net
        seg_net_size = self.seg_net_size
        ost =self.state.node.origin_start
        mask_st = [st[0]-ost[0], st[1]-ost[1], st[2]-ost[2]]

        with torch.no_grad():
            raw_patch = self.raw [st[0]: st[0] + sz[0], 
                                st[1]: st[1] + sz[1],
                                st[2]: st[2] + sz[2]]
            raw_patch = resize (raw_patch, seg_net_size, order=0, mode='reflect', preserve_range=True)
            raw_patch = np.expand_dims (np.expand_dims (raw_patch, 0), 0) # (1, 1, H, W)
            new_mask = net (torch.tensor (raw_patch, device=self.device, dtype=torch.float32))
            new_mask = np.squeeze (new_mask.cpu ().numpy()) * 255

        new_mask = resize (new_mask, sz, order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        cur_mask = self.mask [mask_st[0]: mask_st[0]+sz[0], 
                mask_st[1]: mask_st[1]+sz[1], 
                mask_st[2]: mask_st[2]+sz[2]]

        # print ('cur_mask_shape', cur_mask.shape, 'new_mask_shape', new_mask.shape)
        # print ('st:', st, 'sz:', sz)
        # print ('mask_st', mask_st, 'origin_start', ost)
        self.mask [mask_st[0]: mask_st[0]+sz[0], 
                mask_st[1]: mask_st[1]+sz[1], 
                mask_st[2]: mask_st[2]+sz[2]] = np.maximum (cur_mask, new_mask)

        # print (np.max (self.mask))

        gt_lbl = self.lbl [st[0]-padding: st[0]+sz[0]+padding, 
                            st[1]-padding: st[1] + sz[1]+padding, 
                            st[2]-padding: st[2] + sz[2]+padding]   

        new_mask = np.pad (new_mask, padding, mode='constant', constant_values=0)

        cell_id = self.get_cell_id (self.lbl [st[0]: st[0]+sz[0], 
                                        st[1]: st[1] + sz[1], 
                                        st[2]: st[2] + sz[2]])
        if cell_id in self.viewed:
            return 0
        self.viewed [cell_id] = True
        score = self.calculate_score (gt_lbl, new_mask, cell_id)
        if score < 0.6:
            score = 0
        return score

    def handle_zoomin (self, action):
        next_node_id = self.state.node.id * TREE_BASE + action
        # Revisit a visited node
        if next_node_id in self.history:
            return self.handle_zoomout ()

        self.history [next_node_id] = {'refined':False, 'zoomed':0}
        self.history [self.state.node.id] ['zoomed'] |= 2 ** action

        # print ('----------------------------Zoomin')
        # print ('old:', self.state.node.start, self.state.node.size)

        if self.state.node.level == MAX_LV - 1:
            # instantly refine inner patch
            self.history [next_node_id]['refined']=True
            # Zoomin, refine then zoomout with no stack operation
            self.state.node.step (action)
            reward = self.refine ()
            self.handle_zoomout (pop_last=False)
            info = {
                'up_level' : False,
                'ale.lives': 1,
                'down_level': False
            }
            done = False
            return self.observation (), reward, done, info

        reward = 0
        done = False
        info = {
            'up_level' : False,
            'ale.lives': 1,
            'down_level': True
        }

        self.stack += [copy.deepcopy (self.state)]
        self.state.node.step (action)
        # print ('current:', self.state.node.start, self.state.node.size)
        
        return self.observation (), reward, done, info

    def handle_refine (self):
        # If trying to re-refine
        if self.history [self.state.node.id]['refined']:
            return self.handle_zoomout ()
        
        reward = self.refine ()
        self.history [self.state.node.id]['refined'] = True
        info = {
            'up_level' : False,
            'ale.lives': 1,
            'down_level': False
        }
        done = False

        return self.observation (), reward, done, info

    def handle_zoomout (self, pop_last=True):
        # print ('----------------------------Zoomout')
        # print ('old:', self.state.node.start, self.state.node.size)
        done = True
        observation = self.observation ()
        reward = 0
        info = {
            'up_level' : True and (self.state.node.level!=0),
            'ale.lives': 1 if (self.state.node.level!=0) else 0,
            'down_level': False
        }
        if len (self.stack) > 0 and pop_last:
            self.stack.pop ()
        self.state.node.step (TREE_BASE)
        # print ('current', self.state.node.start, self.state.node.size)
        return observation, reward, done, info

    def step (self, action):
        done = False
        state = copy.deepcopy (self.state)
        self.set_state (state)
        # print ("DEBUG", TREE_BASE)
        prev_state = copy.deepcopy (state)

        info = {'ale.lives': 1}

        if (0 <= action < TREE_BASE):
            return self.handle_zoomin (action)

        if (action == TREE_BASE):
            return self.handle_refine ()

        if (action == TREE_BASE + 1):
            return self.handle_zoomout ()

    def cell_count (self):
        st = self.state.node.start
        sz = self.state.node.size
        gt_lbl = self.lbl [st[0]: st[0]+sz[0], 
                            st[1]: st[1] + sz[1], 
                            st[2]: st[2] + sz[2]]  
        return len (np.unique (gt_lbl))

    def debug (self):
        print ('------------------------DEBUG')
        print (self.state.node.id)
        print (self.history [self.state.node.id]['refined'])
        print (self.history [self.state.node.id]['zoomed'])
        print ('-----------------------------')

    def sample_action (self):
        return self.action_space.sample_action ()
    
    def get_zoomed_mask (self, zoomed):
        # print ("------------------Get Zoomed Mask")
        ret = np.zeros (RL_NET_SIZE[:3], dtype=np.uint8)

        for i in range (TREE_BASE):
            if ((2 ** i) & zoomed) != 0:
                # print ('i', i)
                z0 = int (DZ [i] * 0.5 * RL_NET_SIZE [0]) + RL_NET_SIZE [0] // 12
                y0 = int (DY [i] * 0.5 * RL_NET_SIZE [1]) + RL_NET_SIZE [1] // 12
                x0 = int (DX [i] * 0.5 * RL_NET_SIZE [2]) + RL_NET_SIZE [2] // 12

                size = [RL_NET_SIZE [0]//3, RL_NET_SIZE [1]//3, RL_NET_SIZE[2]//3]
                # print ('size', size)
                # print ((z0, y0, x0)) 
                ret [z0:z0+size[0], y0:y0+size[1], x0:x0+size[2]] = 255
        return ret

    def set_state (self, state):
        self.state = copy.deepcopy (state)

    def observation (self):
        '''
            Observation of size RL_NET_SIZE
            # Raw, mask, zoomed_mask, refined_mask
        '''
        st = self.state.node.start
        sz = self.state.node.size
        ost =self.state.node.origin_start
        mask_st = [st[0]-ost[0], st[1]-ost[1], st[2]-ost[2]]

        raw_patch = self.raw [st[0]: st[0] + sz[0], st[1]: st[1] + sz[1], st[2]: st[2] + sz[2]]

        mask_patch = self.mask [mask_st[0]: mask_st[0] + sz[0], 
                                mask_st[1]: mask_st[1] + sz[1], 
                                mask_st[2]: mask_st[2] + sz[2]]
        if (self.history[self.state.node.id]['refined']):
            refined_mask = np.ones (RL_NET_SIZE[:3], dtype=np.uint8) * 255
        else:
            refined_mask = np.zeros (RL_NET_SIZE[:3], dtype=np.uint8)

        raw_patch = resize (raw_patch, RL_NET_SIZE [:3], order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        mask_patch = resize (mask_patch, RL_NET_SIZE [:3], order=0, mode='reflect', preserve_range=True).astype (np.uint8)

        zoomed_mask = self.get_zoomed_mask (self.history[self.state.node.id]['zoomed'])

        raw_patch = np.expand_dims (raw_patch, -1)
        mask_patch = np.expand_dims (mask_patch, -1)
        refined_mask = np.expand_dims (refined_mask, -1)
        zoomed_mask = np.expand_dims (zoomed_mask, -1)

        ret = np.concatenate ([raw_patch, mask_patch, refined_mask, zoomed_mask], -1)
        return ret

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

        size = ret.shape

        z = size [0] // 2
        y = size [1] // 2
        x = size [2] // 2
        yx = self.concat_last_dim_2_x (ret [z, :, :, :])
        zx = self.concat_last_dim_2_x (ret [:, y, :, :])
        zy = self.concat_last_dim_2_x (ret [:, :, x, :])
        # Concatenate to Y dim
        ret = np.concatenate ([yx, zx, zy], 0)
        ret = np.repeat (np.expand_dims (ret, -1), 3, axis=-1)

        return ret.astype (np.uint8)






