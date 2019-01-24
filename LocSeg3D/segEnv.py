import numpy as np 
from skimage.measure import label
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import adjusted_rand_score

import skimage.io as io
from img_aug_func import *
import copy
import math
from Seg_net.Unet import *
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym

import gym.spaces

MAX_LV = 3 # 0, 1, 2
RL_CHANNELS = 4 # raw, mask, refine?, zoomed_patch?
SEG_CHANNELS = 2
RL_NET_SIZE = (128, 128, RL_CHANNELS)
ORIGIN_SIZE = (512, 512)

DZ = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5]
DY = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5]
DX = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.5]

TREE_BASE = len (DZ)
NUM_ACTIONS = len (DZ) + 2 # n zoom patches, refine?, back 

SIZE_DECAY = 0.6
CELL_THRESHOLD = 200

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
    def __init__ (self):
        self.id = 1
        self.start = [0, 0]
        self.size = list (ORIGIN_SIZE)
        self.level = 0
        self.his = []

    def step (self, action):
        start, size = self.start, self.size
        # Trying to visit child
        if action < TREE_BASE:

            self.id = self.id * TREE_BASE + action
            self.his += [copy.deepcopy ((start, size))]

            #Update window position and size
            start [0] += int (DY [action] * (1 - SIZE_DECAY) * self.size [0])
            start [1] += int (DX [action] * (1 - SIZE_DECAY) * self.size [1])

            size [0] = int (size[0] * SIZE_DECAY)
            size [1] = int (size[1] * SIZE_DECAY)

            self.start, self.size = start, size
            self.level += 1
        else:
            if self.level == 0:
                return
            self.id = self.id // TREE_BASE
            self.start, self.size = self.his.pop ()
            self.level -= 1

class State:
    def __init__ (self, img_id, node):
        self.node = node
        self.img_id = img_id
        self.action_his = []

    def debug (self):
        print ('id:', self.id, 'start:', self.start, 'size:', self.size,
            'target:', self.target, 'depth:', self.depth)

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
    def __init__ (self, raw, lbl, SEG_checkpoints_paths, ds_factors):
        self.action_space = ActionSpace (NUM_ACTIONS)
        self.observation_space = ObservationSpace (RL_NET_SIZE)
        self.obs_shape = RL_NET_SIZE
        self.raw = raw
        self.lbl = lbl

        self.base_start = (0, 0, 0)
        self.base_size = ORIGIN_SIZE
        self.thres = 128
        self.metric = rand_score

        self.rng = np.random.RandomState(time_seed ())
        self.device = torch.device("cuda:1")
        self.setup_nets (SEG_checkpoints_paths, ds_factors)

    def setup_nets (self, SEG_checkponts_paths, ds_factors):
        self.nets = []
        for path in SEG_checkponts_paths:
            net = LightVnet (SEG_CHANNELS, 1).to (self.device)
            checkpoint = torch.load  (path)
            net.load_state_dict (checkpoint['state_dict'])
            self.nets += [net]    

        self.seg_net_sizes = []
        seg_size = list (self.base_size)
        square_size = [64, 128, 256, 512]
        for i in range (len (SEG_checkponts_paths)):
            size = (math.ceil (seg_size[0] * ds_factors[i]), math.ceil (seg_size[1] * ds_factors[i]))
            for sq_size in square_size:
                if sq_size >= size[0]:
                    size = (sq_size, sq_size)
            self.seg_net_sizes += [size]
            seg_size [0] = math.ceil (seg_size[0] * SIZE_DECAY)
            seg_size [1] = math.ceil (seg_size[1] * SIZE_DECAY)


    def reset (self):
        img_id = self.rng.randint (0, len (self.raw))
        self.state = State (img_id, Node ())
        self.mask = np.zeros (self.raw [0].shape, dtype=np.uint8)
        self.history = {}
        self.history [self.state.node.id] = {'refined':False, 'zoomed':0}
        self.stack = []
        return self.observation ()

    def refine (self):
        st = self.state.node.start
        sz = self.state.node.size
        img_id = self.state.img_id
        net = self.nets[self.state.node.level]
        seg_net_size = self.seg_net_sizes [self.state.node.level]


        with torch.no_grad():
            raw_patch = self.raw [img_id, st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]
            raw_patch = resize (raw_patch, seg_net_size, order=0, mode='reflect', preserve_range=True)
            old_mask_patch = self.mask [st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]
            old_mask_patch = resize (old_mask_patch, seg_net_size, order=0, mode='reflect', preserve_range=True)

            raw_patch = np.expand_dims (np.expand_dims (raw_patch, 0), 0) # (1, 1, H, W)
            old_mask_patch = np.expand_dims (np.expand_dims (old_mask_patch, 0), 0) # (1, 1, H, W)

            #append old mask
            x = np.concatenate ([raw_patch, old_mask_patch], 1) #black out for current situation # (1, 2, H, W)

            new_mask = net (torch.tensor (x, device=self.device, dtype=torch.float32))
            new_mask = np.squeeze (new_mask.cpu ().numpy()) * 255

        new_mask = resize (new_mask, sz, order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        cur_mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1]]
        gt_lbl = self.lbl [self.state.img_id, st[0]: st[0]+sz[0], 
                            st[1]: st[1] + sz[1]]

        old_score = self.metric (gt_lbl, cur_mask)

        self.mask [st[0]: st[0]+sz[0], 
                    st[1]: st[1] + sz[1]] = new_mask

        score = self.metric (gt_lbl, new_mask)
        return score - old_score

    def cal_metric (self):
        st = self.state.node.start
        sz = self.state.node.size
        img_id = self.state.img_id

        cur_mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1]]

        gt_lbl = self.lbl [self.state.img_id, st[0]: st[0]+sz[0], 
                            st[1]: st[1] + sz[1]]

        return self.metric (gt_lbl, cur_mask)


    def handle_zoomin (self, action):
        next_node_id = self.state.node.id * TREE_BASE + action
        # Revisit a visited node
        if next_node_id in self.history:
            return self.handle_zoomout ()

        self.history [next_node_id] = {'refined':False, 'zoomed':0}
        self.history [self.state.node.id] ['zoomed'] |= 2 ** action

        if self.state.node.level == MAX_LV - 1:
            # instantly refine inner patch
            old_score = self.cal_metric ()
            self.history [next_node_id]['refined']=True
            self.state.node.step (action)
            self.refine ()
            self.handle_zoomout (pop_last=False)
            info = {
                'up_level' : False,
                'ale.lives': 1,
                'down_level': False
            }
            new_score = self.cal_metric ()
            reward = new_score - old_score
            done = False
            return self.observation (), reward, done, info

        reward = 0
        done = False
        info = {
            'current_score' : self.cal_metric (),
            'up_level' : False,
            'ale.lives': 1,
            'down_level': True
        }

        self.stack += [copy.deepcopy (self.state)]
        self.state.node.step (action)
        
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
        return observation, reward, done, info

    def step (self, action):
        done = False
        state = copy.deepcopy (self.state)
        self.set_state (state)

        prev_state = copy.deepcopy (state)

        info = {'ale.lives': 1}

        if (0 <= action < TREE_BASE):
            return self.handle_zoomin (action)

        if (action == TREE_BASE):
            return self.handle_refine ()

        if (action == TREE_BASE + 1):
            return self.handle_zoomout ()

    def sample_action (self):
        return self.action_space.sample_action ()
    
    def get_zoomed_mask (self, zoomed):
        ret = np.zeros (RL_NET_SIZE[:2], dtype=np.uint8)

        for i in range (TREE_BASE):
            if ((2 ** i) & zoomed) != 0:
                y0 = int (DY [i] * 0.75 * RL_NET_SIZE [0])
                x0 = int (DX [i] * 0.75 * RL_NET_SIZE [1])

                size = [RL_NET_SIZE [0]//4, RL_NET_SIZE [1]//4]
                ret [y0:y0+size[0], x0:x0+size[1]] = 255
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
        img_id = self.state.img_id

        

        raw_patch = self.raw [img_id, st[0]: st[0] + sz[0], st[1]: st[1] + sz[1]]

        mask_patch = self.mask [st[0]: st[0] + sz[0], st[1]: st[1] + sz[1]]
        if (self.history[self.state.node.id]['refined']):
            refined_mask = np.ones (RL_NET_SIZE[:2], dtype=np.uint8) * 255
        else:
            refined_mask = np.zeros (RL_NET_SIZE[:2], dtype=np.uint8)

        raw_patch = resize (raw_patch, RL_NET_SIZE [:2], order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        mask_patch = resize (mask_patch, RL_NET_SIZE [:2], order=0, mode='reflect', preserve_range=True).astype (np.uint8)

        zoomed_mask = self.get_zoomed_mask (self.history[self.state.node.id]['zoomed'])

        raw_patch = np.expand_dims (raw_patch, -1)
        mask_patch = np.expand_dims (mask_patch, -1)
        refined_mask = np.expand_dims (refined_mask, -1)
        zoomed_mask = np.expand_dims (zoomed_mask, -1)

        ret = np.concatenate ([raw_patch, mask_patch, refined_mask, zoomed_mask], -1)
        return ret

    def get_gt_label_RGB (self):
        st = self.state.node.start
        sz = self.state.node.size
        img_id = self.state.img_id

        gt_lbl = self.lbl [self.state.img_id, st[0]: st[0]+sz[0], 
                            st[1]: st[1] + sz[1]]
        gt_lbl = resize (gt_lbl, RL_NET_SIZE[:2], order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        ret = gt_lbl.flatten ().tolist ()
        ret = list (map (index2rgb, ret))
        ret = np.array (ret, dtype=np.uint8)
        ret = ret.reshape (RL_NET_SIZE[:2] + (3,))
        return ret

    def get_labeled_mask_RGB (self):
        st = self.state.node.start
        sz = self.state.node.size
        mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1]]

        labeled_mask = label (mask > CELL_THRESHOLD, background=0)
        labeled_mask = resize (labeled_mask, RL_NET_SIZE[:2], order=0, mode='reflect', preserve_range=True).astype (np.uint8)
        ret = labeled_mask.flatten ().tolist ()
        ret = list (map (index2rgb, ret))
        ret = np.array (ret, dtype=np.uint8)
        ret = ret.reshape (RL_NET_SIZE[:2] + (3,))
        return ret

    def render (self):
        '''
            return img of shape (H, W, 3)
        '''
        ret = self.observation ();

        raw_patch = ret [...,0:1]
        mask_patch = ret [...,1:2]
        refined_mask = ret [...,2:3]
        zoomed_mask = ret [...,3:4]

        gt_lbl = self.get_gt_label_RGB ()
        lbled_mask = self.get_labeled_mask_RGB ()

        ret = np.concatenate ([raw_patch, mask_patch, refined_mask, zoomed_mask], 1)
        ret = np.repeat (ret, 3, axis=-1)

        ret = np.concatenate ([ret, gt_lbl, lbled_mask], 1)

        return ret






