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
from malis import rand_index 
from random import shuffle
from PIL import Image, ImageFilter
from utils import guassian_weight_map, density_map
from skimage.draw import line_aa
from misc.Voronoi import *

debug = True

class General_env (gym.Env):
    def init (self, config):
        self.T = config ['T']
        self.r = config ["radius"]
        self.size = config ["size"]
        self.speed = config ["speed"]
        if config ["use_lbl"]:
            self.observation_space = Box (0, 1.0, shape=[2] + self.size, dtype=np.float32)
        else:
            self.observation_space = Box (-1.0, 1.0, shape=[self.T + 1] + self.size, dtype=np.float32)
        self.rng = np.random.RandomState(time_seed ())
        self.max_lbl = 2 ** self.T - 1

    def step (self, action):
        self.new_lbl = self.lbl * 2 + action
        done = False
        if self.config["reward"] == "gaussian":
            w_map = guassian_weight_map ((self.r*2+1, self.r*2+1))
            reward = self.split_penalty_step (w_map=w_map)
            reward += self.split_reward_step (w_map=w_map)
        elif self.config ["reward"] == "density":
            density = density_map (self.gt_lbl)
            reward = self.split_penalty_step (density=density)
            reward += self.split_reward_step (density=density)
        else:
            reward = self.split_penalty_step ()
            reward += self.split_reward_step ()

        self.lbl = self.new_lbl
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255

        self.step_cnt += 1
        if self.step_cnt >= self.T:
            done = True
        info = {}
        self.rewards.append (reward)
        self.sum_reward += reward
        return self.observation (), reward, done, info

    def split_penalty_step (self, w_map=None, density=None):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        if density is not None:
            density_cp = np.pad (density, self.r, 'constant', constant_values=0.33)
            norm_map = np.zeros_like (reward)

        first_step = np.max (lbl_cp) < 1
        r = self.r
        # Wrongly different label with neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_old = self.lbl == lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                tmp = (I != I_hat) & (I_hat == True) & ((I_old == I_hat) | first_step)
                if w_map is not None:
                    reward += tmp.astype (np.float32) * w_map [yr + r, xr + r]
                elif density is not None:
                    reward += tmp * density * density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                    norm_map += density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                else:
                    reward += tmp
        reward *= -1
        if w_map is not None:
            reward = reward / np.sum (w_map)
        elif density is not None:   
            reward = reward / norm_map
        else:
            reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def split_reward_step (self, w_map=None, density=None):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        if density is not None:
            density_cp = np.pad (density, self.r, 'constant', constant_values=0.33)
            norm_map = np.zeros_like (reward)
        r = self.r
        first_step = np.max (lbl_cp) < 1
        # Correctly different label with neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_old = self.lbl == lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                tmp = (I == I_hat) & (I_hat == False) & ((I_old != I_hat) | first_step)
                if w_map is not None:
                    reward += tmp.astype (np.float32) * w_map [yr + r, xr + r]
                elif density is not None:
                    reward += tmp * density * density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                    norm_map += density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                else:
                    reward += tmp 
                 
        if w_map is not None:
            reward = reward / np.sum (w_map)
        elif density is not None:
            reward = reward / norm_map
        else:
            reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def observation (self):
        if not self.config ["use_lbl"]:
            obs = np.concatenate ([
                    self.raw [None] * 2 - 255.0,
                    self.mask,
                ], 0)

        else:
            lbl = self.lbl / self.max_lbl * 255.0
            obs = np.concatenate ([
                    self.raw [None].astype (np.float32),
                    lbl [None]
                ])
        if self.obs_format == "CHW":
            ret = obs.astype (np.float32) / 255.0
            return ret 
        else:
            ret = np.transpose (obs, [1, 2, 0]) / 255.0
            return ret

    def render (self):
        raw = np.repeat (np.expand_dims (self.raw, -1), 3, -1).astype (np.uint8)
        lbl = self.lbl.astype (np.int32)
        lbl = lbl2rgb (lbl)
        gt_lbl = lbl2rgb (self.gt_lbl)
        
        masks = []
        for i in range (self.T):
            mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        rewards = []
        for reward_i in [self.sum_reward] + self.rewards:
            reward_i = ((reward_i + 1) / 2 * 255).astype (np.uint8)
            reward_i = np.repeat (np.expand_dims (reward_i, -1), 3, -1).astype (np.uint8)   
            rewards.append (reward_i)
        while (len (rewards) < self.T + 1):
            rewards.append (np.zeros_like (rewards [0]))

        line1 = [raw, lbl,gt_lbl, ] + masks
        while (len (rewards) < len (line1)):
            rewards = [np.zeros_like (rewards [-1])] + rewards
        line1 = np.concatenate (line1, 1)
        line2 = np.concatenate (rewards, 1)
        ret = np.concatenate ([line1, line2], 0)

        return ret

class Voronoi_env (General_env):
    def __init__ (self, config, obs_format="CHW"):
        self.config = config
        self.obs_format = obs_format
        self.type = "train"
        self.init (config)

    def init (self, config):
        super ().init (config)
        self.num_segs = config ["num_segs"]

    def reset (self):
        self.step_cnt = 0
        if not debug:
            self.raw = generate_voronoi_diagram (self.config ["size"][0], self.config ["size"][1], self.num_segs, self.rng)
        else:
            size = self.size
            self.raw = np.zeros (self.size)
            half = self.size [0] // 2
            self.raw [2:half, 2:half] = 1
            self.raw [2:half, half:size[1]] = 2
            self.raw [half:size[0], 2:half] = 3
            self.raw [half:size[0], half:size[1]] = 4

        prob = get_boudary (self.raw [None], 0, 0) [0].astype (np.float32)
        self.gt_lbl = label (prob > 128).astype (np.int32)
        # self.gt_lbl = self.raw.astype (np.int32)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.raw = prob
        self.rewards = []
        return self.observation ()


class EM_env (General_env):
    def __init__ (self, raw_list, config, type, gt_lbl_list=None, obs_format="CHW"):
        self.type = type
        self.raw_list = raw_list.astype (np.float32)
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(time_seed ())
        self.config = config
        self.obs_format = obs_format
        self.init (config)

    def random_crop (self, size, imgs):
        y0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
        x0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
        ret = []
        for img in imgs:
            ret += [img[y0:y0+size[0], x0:x0+size[1]]]
        return ret

    def reset (self):
        self.step_cnt = 0
        z0 = self.rng.randint (0, len (self.raw_list))
        self.raw = copy.deepcopy (self.raw_list [z0])
        if (self.gt_lbl_list is not None):
            self.gt_lbl = copy.deepcopy (self.gt_lbl_list [z0])
        else:
            self.gt_lbl = np.zeros_like (self.raw)

        self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])

        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []
        return self.observation ()



