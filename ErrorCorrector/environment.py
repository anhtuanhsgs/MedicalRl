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


def rand_score (gt_lbl, pred_lbl):
    ret = adjusted_rand_score (pred_lbl.flatten (), gt_lbl.flatten ())
    print (type (ret))
    return ret

def malis_rand_index (gt_lbl, pred_lbl):
    ret = rand_index (gt_lbl, pred_lbl) [0]
    ret = float (ret)
    return ret

def malis_f1_score (gt_lbl, pred_lbl):
    ret = rand_index (gt_lbl, pred_lbl) [1]
    ret = float (ret)
    return ret

class EM_env (gym.Env):
    def __init__ (self, raw_list, lbl_list, cell_prob_list, config, type, gt_lbl_list=None, obs_format="CHW"):
        self.type = type
        self.raw_list = raw_list.astype (np.float32)
        self.lbl_list = lbl_list.astype (np.float32)
        self.cell_prob_list = cell_prob_list.astype (np.float32)
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(time_seed ())
        self.config = config
        self.obs_format = obs_format
        self.init (config)

    def init (self, config):
        self.corrector_size = config ['corrector_size']
        self.spliter = config ['spliter']
        self.merger = config ['merger']
        self.cell_thres = config ['cell_thres']
        self.T = config ['T']
        self.agent_out_shape = config ['agent_out_shape']
        self.observation_shape = config ['observation_shape']
        if self.obs_format == "HWC":
            self.observation_shape = self.observation_shape [2:] + self.observation_shape [:2]
        # size = self.raw_list [0].shape
        self.action_space = Discrete(np.prod (self.agent_out_shape))
        self.observation_space = Box (0.0, 255.0, shape=(config ['num_feature'], 
                            self.observation_shape[1], self.observation_shape[2]), dtype=np.float32)
        self.gpu_id = config ['env_gpu']
        
        self.metric = malis_f1_score 
        self.valid_range = [
                [self.corrector_size [0] // 2, self.observation_shape[1] - self.corrector_size [0] // 2],
                [self.corrector_size [1] // 2, self.observation_shape[2] - self.corrector_size [1] // 2]
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

        self.raw, self.lbl, self.prob, self.gt_lbl = self.random_crop (self.observation_shape [1:],
                        [self.raw, self.lbl, self.prob], mask=self.gt_lbl)
        self.lbl = self.shuffle_lbl (self.lbl.astype (np.int32))
        self.info_mask = np.zeros_like (self.raw)

        if self.type == 'train':
            self.old_score = self.metric (self.gt_lbl, self.lbl.astype (np.uint32))
            # print ('current_score:', self.old_score)
        else:
            self.old_score = 0
        self.lbl = self.transform_lbl (self.lbl.astype (np.float32))
        return self.observation ()

    def crop_center (self, center, imgs, size):
        y0 = center [0] - size [0] // 2
        x0 = center [1] - size [1] // 2
        # print ('crop center', center, y0, x0)
        ret = []
        for img in imgs:
            ret += [img [y0:y0+size[0], x0:x0+size[1]]]
        return ret

    def random_crop (self, size, images, mask=None):
        stack = []
        for img in images:
            stack += [np.expand_dims (img, -1)]
        stack = np.concatenate (stack, -1)
        cropper = A.RandomCrop (height=size [0], width=size[1], p=1)
        cropped = cropper (image=stack, mask=mask)
        cropped_stack, mask = cropped ['image'], cropped ['mask']
        ret = []
        for i in range (len (images)):
            ret += [cropped_stack [..., i]]
        ret += [mask]
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
        ESP = 1e-6
        if (np.max (lbl) < ESP):
            self.max_lbl = 0.0
            return lbl
        self.max_lbl = np.max (lbl)
        return lbl / np.max (lbl) * 255.0

    def shuffle_lbl (self, lbl):
        per = list (range (1, 1000))
        shuffle(per)
        per = [0] + per
        vf = np.vectorize (lambda x: per[x])
        return vf (lbl)

    def observation (self):
        # lbl = (self.lbl * self.max_lbl / 255.0).astype (np.int32)
        # lbl = np.transpose (lbl2rgb (lbl), [2, 0, 1])
        lbl = np.repeat (self.zeros_like (self.lbl) [None], 3, 0)
        obs = np.concatenate ([
                self.raw [None],
                lbl,
                self.prob [None],
                self.info_mask [None]
            ], 0)
        if self.obs_format == "CHW":
            return obs.astype (np.float32) / 255.0
        else:
            obs = np.transpose (obs, [1, 2, 0]) / 255.0
            return obs

    def step (self, action):
        assert (action < np.prod (self.agent_out_shape))
        self.step_cnt += 1
        # if (self.args ["Continuous"]):
        #     error_index = 
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
            patches = self.crop_center (error_index, [self.raw, self.lbl, self.prob, self.info_mask, self.gt_lbl], self.corrector_size)
        else:
            patches = self.crop_center (error_index, [self.raw, self.lbl, self.prob, self.info_mask], self.corrector_size)

        new_prob = corrector (patches [0], patches [2], gpu_id=self.gpu_id)
        patches [2][::] = new_prob
        new_label = label (self.prob > self.cell_thres, background=0).astype (np.int32)
        if self.type == 'train':
            new_score = self.metric (self.gt_lbl, new_label)
            reward = (new_score - self.old_score) * 10
            self.old_score = new_score
            # print ('current score:', self.old_score)
        else:
            reward = 0
        patches [3][::] = int (1.0 * self.step_cnt / self.T * 255.0)
        new_label = self.shuffle_lbl (new_label)
        self.lbl [::,::] = self.transform_lbl (new_label.astype (np.float32))


        if (self.step_cnt >= self.T):
            reward += self.old_score * 10
            done = True
        else:
            done = False
        info = {}
        return self.observation (), reward, done, info

    def render (self):
        raw = np.repeat (np.expand_dims (self.raw, -1), 3, -1).astype (np.uint8)
        prob = np.repeat (np.expand_dims (self.prob, -1), 3, -1).astype (np.uint8)
        info_mask = np.repeat (np.expand_dims (self.info_mask, -1), 3, -1).astype (np.uint8)
        # print ("max :", self.max_lbl)
        lbl = (self.lbl * self.max_lbl / 255.0).astype (np.int32)
        # plt.imshow (lbl)
        # plt.show ()
        # print ('current rand_index:', self.metric (self.gt_lbl, lbl))
        lbl = lbl2rgb (lbl)
        gt_lbl = lbl2rgb (self.gt_lbl)

        ret = np.concatenate ([raw,
                prob,
                lbl,
                gt_lbl,
                info_mask
            ], 1)

        return ret