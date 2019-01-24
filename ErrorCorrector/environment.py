import os, sys, glob, time, copy
from os import sys, path
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box
from skimage.measure import label
from sklearn.metrics import adjusted_rand_score
from cv2 import resize
from Utils.utils import *
from Utils.img_aug_func import *
import albumentations as A
import random
from gym.spaces import Box, Discrete, Tuple
import matplotlib.pyplot as plt


def rand_score (gt_lbl, mask):
    lbled_mask = label (mask, background=0)
    return adjusted_rand_score (lbled_mask.flatten (), gt_lbl.flatten ())

class EM_env (gym.Env):
    def __init__ (self, raw_list, lbl_list, cell_prob_list, config, type, gt_lbl_list=None):
        self.type = type
        self.raw_list = raw_list.astype (np.float32)
        self.lbl_list = lbl_list.astype (np.float32)
        self.cell_prob_list = cell_prob_list.astype (np.float32)
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(time_seed ())
        self.config = config
        self.init (config)

    def init (self, config):
        self.corrector_size = config ['corrector_size']
        self.spliter = config ['spliter']
        self.merger = config ['merger']
        self.cell_thres = config ['cell_thres']
        self.T = config ['T']
        self.agent_out_shape = config ['agent_out_shape']
        size = self.raw_list [0].shape
        self.action_space = Discrete(np.prod (self.agent_out_shape))
        self.observation_space = Box (0.0, 255.0, shape=(config ['num_feature'], 
                            size[0], size[1]), dtype=np.float32)

        
        self.metric = rand_score
        self.valid_range = [
                [self.corrector_size [0] // 2, size [0] - self.corrector_size [0] // 2],
                [self.corrector_size [1] // 2, size [1] - self.corrector_size [1] // 2]
            ]
        # print ('valid range', self.valid_range)

    def reset (self):
        self.step_cnt = 0
        z0 = self.rng.randint (0, len (self.raw_list))
        self.raw = copy.deepcopy (self.raw_list [z0])
        self.lbl = copy.deepcopy (self.lbl_list [z0])
        self.prob = copy.deepcopy (self.cell_prob_list [z0])
        # print (np.max (self.raw), np.max (self.lbl), np.max (self.prob))
        # self.raw, self.lbl, self.prob = self.aug (self.raw, self.lbl, self.prob)

        if (self.gt_lbl_list is not None):
            self.gt_lbl = copy.deepcopy (self.gt_lbl_list [z0])

        if self.type == 'train':
            self.old_score = self.metric (self.gt_lbl, self.lbl.astype (np.uint32))
            # print ('current_score:', self.old_score)
        else:
            self.old_score = 0
        self.lbl = self.transform_lbl (self.lbl)
        return self.observation ()

    def crop_center (self, center, imgs, size):
        y0 = center [0] - size [0] // 2
        x0 = center [1] - size [1] // 2
        # print ('crop center', center, y0, x0)
        ret = []
        for img in imgs:
            ret += [img [y0:y0+size[0], x0:x0+size[1]]]
        return ret

    def int2index (self, x, size):
        ret = ()
        for l in size [::-1]:
            ret += (x % l,)
            x = x // l
        return ret [::-1]

    def index2validrange (self, idx, size):
        idx_ret = []
        for i in range (len (idx)):
            idx_ret += [int (idx[i] / (size [i] - 1) * (self.valid_range [i][1] - self.valid_range [i][0]) + self.valid_range [i][0])]
        return idx_ret

    def aug (self, raw, lbl, mask):
        paired = np.concatenate ([
            np.expand_dims (lbl, -1),
            np.expand_dims (mask, -1)
        ], -1)
        aug = A.Compose ([
            A.HorizontalFlip (),
            A.RandomRotate90 (p=0.5),
            A.VerticalFlip (),
            A.Transpose (),
            A.RandomGamma(p=1, gamma_limit=(30, 236)),
            A.RandomContrast(p=0.8),
            A.GaussNoise (p=0.5),
            A.Blur (p=0.5)
        ], p=0.8)

        ret = aug (image=raw, mask=paired)
        raw, paired = ret ['image'], ret['mask']
        return raw, paired [...,0], paired [..., 1]

    def transform_lbl (self, lbl):
        return lbl / np.max (lbl) * 255.0

    def observation (self):
        obs = np.concatenate ([
                self.raw [None],
                self.lbl [None],
                self.prob [None]
            ])
        return obs

    def step (self, action):
        self.step_cnt += 1
        action_index = self.int2index (action, self.agent_out_shape)
        error_index = self.index2validrange (action_index [1:], self.agent_out_shape [1:])

        # print ('valid range: ', self.valid_range, 'error index: ', action_index [1:])
        # print ('action index: ', action_index)
        # print ('error index', error_index)
        # print ('corrector size', self.corrector_size)
        if action_index [0] == 0:
            corrector = self.spliter
        else:
            corrector = self.merger
        if self.type == 'train':
            patches = self.crop_center (error_index, [self.raw, self.lbl, self.prob, self.gt_lbl], self.corrector_size)
        else:
            patches = self.crop_center (error_index, [self.raw, self.lbl, self.prob], self.corrector_size)

        new_prob = corrector (patches [0], patches [2])
        patches [2][::] = new_prob
        new_label = label (self.prob > self.cell_thres)
        if self.type == 'train':
            new_score = self.metric (self.gt_lbl, new_label)
            reward = new_score - self.old_score
            self.old_score = new_score
            # print ('current score:', self.old_score)
        else:
            reward = 0

        
        self.lbl [::,::] = self.transform_lbl (new_label.astype (np.float32))
        if (self.step_cnt >= self.T):
            done = True
        else:
            done = False
        info = {}
        return self.observation (), reward, done, info

    def render (self):
        raw = (np.repeat (np.expand_dims (self.raw, -1), 3, -1) * 255).astype (np.uint8)
        prob = (np.repeat (np.expand_dims (self.prob, -1), 3, -1) * 255).astype (np.uint8)
        lbl = (1.0 / self.lbl).astype (np.uint32)
        lbl = lbl.flatten ().tolist ()
        lbl = list (map (index2rgb, lbl))
        lbl = np.array (lbl, dtype=np.uint8)

        ret = np.concatenate ([raw,
                lbl,
                self.prob
            ], 1)

        return ret