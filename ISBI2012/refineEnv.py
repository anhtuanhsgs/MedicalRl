import numpy as np 
from skimage.measure import label
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import adjusted_rand_score

import skimage.io as io
from img_aug_func import *
import os
import copy
import math
from FusionNet.FusionNet import *
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym

import albumentations as A

import gym.spaces

MAX_LV = 3 # 0, 1, 2
RL_CHANNELS = 4 # raw, mask, refine?, zoomed_patch?
SEG_CHANNELS = 2
RL_NET_SIZE = (256, 256, RL_CHANNELS)
ORIGIN_SIZE = (512, 512)
TREE_BASE = 5
NUM_ACTIONS = 5 + 2 # 5 zoom patches, refine?, back 
DY = [0., 0., 1., 1., .5]
DX = [0., 1., 0., 1., .5]
FEATURES = [16, 32, 64, 128]
SIZE_DECAY = 0.6
CELL_THRESHOLD = 10
REWARD_THRESHOLD = 0.001

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
    def __init__ (self, start=[0, 0]):
        self.id = 1
        self.start = copy.deepcopy (start)
        self.size = list (ORIGIN_SIZE)
        self.level = 0
        self.his = []

    def step (self, action):
        start, size = self.start, self.size
        # Trying to visit child
        if action < TREE_BASE:

            self.id = self.id * TREE_BASE + action
            self.his += [copy.deepcopy ((start, size))]

            old_size = copy.deepcopy (size)
            size [0] = math.ceil (size[0] * SIZE_DECAY / 16) * 16
            size [1] = math.ceil (size[1] * SIZE_DECAY / 16) * 16
            #Update window position and size
            start [0] += int (DY [action] * (old_size [0] - size[0]))
            start [1] += int (DX [action] * (old_size [1] - size[1]))

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
    def __init__ (self, raw, lbl, SEG_checkpoints_paths, env_type='train', training=True):
        self.action_space = ActionSpace (NUM_ACTIONS)
        self.observation_space = ObservationSpace (RL_NET_SIZE)
        self.obs_shape = RL_NET_SIZE
        self.raw_origin = raw
        self.lbl_origin = lbl
        self.training = training
        self.base_start = (0, 0, 0)
        self.base_size = ORIGIN_SIZE
        self.thres = 128
        self.metric = rand_score
        self.env_type = env_type

        self.rng = np.random.RandomState(time_seed ())
        self.device = torch.device("cuda:1")
        self.setup_nets (SEG_checkpoints_paths)

    def setup_nets (self, SEG_checkponts_paths):
        self.nets = []
        self.optimizers = []
        self.lr_schedulers = []
        self.loss_func = nn.BCELoss ()
        self.optim_steps = [0] * len (SEG_checkponts_paths) 
        self.save_period = 1000
        for path in SEG_checkponts_paths:
            net = FusionNet (SEG_CHANNELS, FEATURES, 1).to (self.device)
            net.share_memory ()
            checkpoint = torch.load  (path)
            net.load_state_dict (checkpoint['state_dict'])
            self.nets += [net]    
            self.optimizers += [optim.Adam (net.parameters (), lr=1e-4)]
            self.lr_schedulers += [optim.lr_scheduler.StepLR (self.optimizers[-1], step_size=100, gamma=0.999)]


        self.seg_net_sizes = []
        seg_size = list (self.base_size)
        print ("Setup nets, SEG net size:")
        for i in range (len (SEG_checkponts_paths)):
            self.seg_net_sizes += [(math.ceil (seg_size[0]), math.ceil (seg_size[1]))]
            seg_size [0] = math.ceil (seg_size[0] * SIZE_DECAY / 16) * 16
            seg_size [1] = math.ceil (seg_size[1] * SIZE_DECAY / 16) * 16
            print (self.seg_net_sizes [-1])

    def aug_reset (self):
        p_rand_start = 0.5
        if self.rng.rand () > p_rand_start:
            return
        self.step (TREE_BASE)
        aug = A.Compose ([
            A.GaussNoise (p=0.5),
            A.MedianBlur (p=0.5, blur_limit=5),
            A.Blur (p=0.5)
        ], p=0.8)
        self.mask = aug (image=self.mask, mask=None) ['image']

    def reset (self):
        img_id = self.rng.randint (0, len (self.raw_origin))
        self.raw = self.raw_origin [img_id]
        self.lbl = self.lbl_origin [img_id]
        self.raw, self.lbl = self.aug ([self.raw, self.lbl])
        y0 = self.rng.randint (0, self.raw_origin.shape[1] - ORIGIN_SIZE [0] + 1)
        x0 = self.rng.randint (0, self.raw_origin.shape[2] - ORIGIN_SIZE [1] + 1)
        self.state = State (img_id, Node ([y0, x0]))
        self.mask = np.zeros (self.raw_origin [0].shape, dtype=np.uint8)
        self.history = {}
        self.history [self.state.node.id] = {'refined':False, 'zoomed':0}
        self.stack = []
        self.aug_reset ()
        return self.observation ()

    def reset_2 (self, img_id, mask, start):
        self.state = State (img_id, Node (start))
        self.raw = self.raw_origin [img_id]
        self.lbl = self.lbl_origin [img_id]
        self.mask = copy.deepcopy (mask)
        self.history = {}
        self.history [self.state.node.id] = {'refined':False, 'zoomed':0}
        self.stack = []
        return self.observation ()

    def aug (self, imgs):
        ret = []
        rotk = self.rng.randint (0, 4)
        flipk = self.rng.randint (1, 5)
        for img in imgs:
            img = np.rot90(img, rotk, axes=(0,1))
            img = random_flip (img, flipk)
            ret += [img.copy ()]
        return ret


    def threshold_reward (self, reward):
        if reward < REWARD_THRESHOLD and reward > 0:
            reward = 0
        return reward

    def save_model (self, level):
        path = 'jointrained_FusionNet/' + str ('level'+str (level)) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        model = self.nets[level]
        optimizer = self.optimizers [level]
        i_iter = self.optim_steps [level]
        state = {
            'i_iter': i_iter,
            'state_dict': model.state_dict (),
            'optimizer': optimizer.state_dict ()
        }
        path += 'checkpoint_' + str (i_iter) + '.pth.tar'
        torch.save (state, path)

    def refine (self):
        st = self.state.node.start
        sz = self.state.node.size
        img_id = self.state.img_id
        net = self.nets[self.state.node.level]
        seg_net_size = self.seg_net_sizes [self.state.node.level]
        optimizer = self.optimizers [self.state.node.level]
        lr_scheduler = self.lr_schedulers [self.state.node.level]
        self.optim_steps [self.state.node.level] += 1

        cur_mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1]]

        gt_lbl = self.lbl [st[0]: st[0]+sz[0], 
                            st[1]: st[1] + sz[1]]

        if self.training:
            raw_patch = self.raw [st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]
            old_mask_patch = self.mask [st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]

            raw_patch = np.expand_dims (np.expand_dims (raw_patch, 0), 0) # (1, 1, H, W)
            old_mask_patch = np.expand_dims (np.expand_dims (old_mask_patch, 0), 0) # (1, 1, H, W)

            #append old mask
            x = np.concatenate ([raw_patch, old_mask_patch], 1)

            new_mask = net (torch.tensor (x, device=self.device, dtype=torch.float32))
            gt_mask_t = torch.tensor ((gt_lbl > 0).astype (np.float32), device=self.device, dtype=torch.float32).unsqueeze (0).unsqueeze (0)

            loss = self.loss_func (new_mask, gt_mask_t)
            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()
            lr_scheduler.step ()

            new_mask = new_mask.detach().cpu ().numpy() [0][0] * 255

            if self.optim_steps [self.state.node.level] % self.save_period == 0:
                self.save_model (self.state.node.level)

        else:
            with torch.no_grad():
                raw_patch = self.raw [st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]
                old_mask_patch = self.mask [st[0]: st[0]+sz[0], st[1]: st[1] + sz[1]]

                raw_patch = np.expand_dims (np.expand_dims (raw_patch, 0), 0) # (1, 1, H, W)
                old_mask_patch = np.expand_dims (np.expand_dims (old_mask_patch, 0), 0) # (1, 1, H, W)

                #append old mask
                x = np.concatenate ([raw_patch, old_mask_patch], 1)

                new_mask = net (torch.tensor (x, device=self.device, dtype=torch.float32))
                gt_mask_t = torch.tensor (gt_lbl > 0, device=self.device, dtype=torch.float32).unsqueeze ().unsqueeze ()

                new_mask = new_mask.cpu ().numpy() [0][0] * 255

        old_score = self.metric (gt_lbl, cur_mask)

        self.mask [st[0]: st[0]+sz[0], 
                    st[1]: st[1] + sz[1]] = new_mask

        score = self.metric (gt_lbl, new_mask)
        reward = score - old_score
        reward = self.threshold_reward (reward)
        return reward

    def cal_metric (self):
        st = self.state.node.start
        sz = self.state.node.size
        img_id = self.state.img_id

        cur_mask = self.mask [st[0]: st[0]+sz[0], 
                        st[1]: st[1] + sz[1]]

        gt_lbl = self.lbl [st[0]: st[0]+sz[0], 
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
            reward = self.threshold_reward (reward)

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

        

        raw_patch = self.raw [st[0]: st[0] + sz[0], st[1]: st[1] + sz[1]]

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

        gt_lbl = self.lbl [st[0]: st[0]+sz[0], 
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






