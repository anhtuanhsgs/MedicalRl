#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import argparse
import cv2
import numpy as np
import tensorflow as tf
import gym

from tensorpack import *
from resnet_model import *

from DQNModel import Model as DQNModel
from common import LogVisualizeEpisode
from expreplay import ExpReplay 
# from atari import AtariPlayer
from natsort import natsorted
import os,sys, argparse, glob
import functools
from utils import setup_data
sys.path.append('../')
from environment import EM_env
from Utils.img_aug_func import *
from Utils.utils import *
from CorrectorModule.corrector_utils import *

UPDATE_FREQ = 4

MEMORY_SIZE = 1e5
INIT_MEMORY_SIZE = 1e5 / 20 
STEPS_PER_EPOCH = 10000 // UPDATE_FREQ  # each epoch is 100k played frames
TARGET_NET_UPDATE = 1000 // UPDATE_FREQ # update target network every 10k steps
LEARNING_RATE = [(60, 4e-4), (100, 2e-4), (500, 5e-5)] 
EXPLORATION = (0, 1), (1, 0.9), (10, 0.9), (20, 0.1), (320, 0.01) 


parser = argparse.ArgumentParser()

parser.add_argument ('--load', help='load model')

parser.add_argument('--env', 
    default='EM_env', 
    metavar='ENV', 
    help='environment to train on (default: EM_env)')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=15,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

parser.add_argument ('--algo', 
    help='algorithm',
    choices=['DQN', 'Double', 'Dueling'], 
    default='Double')

parser.add_argument (
    '--spliter',
    default='FusionNet',
    metavar='SPL',
    choices=['FusionNet', 'Thres'])

parser.add_argument (
    '--merger',
    default='FusionNet',
    metavar='MER',
    choices=['FusionNet', 'Thres'])

parser.add_argument(
    '--save-period',
    type=int,
    default=1,
    metavar='SP',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--log-period',
    type=int,
    default=1,
    metavar='LP',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--batch-size',
    type=int,
    default=12,
    metavar='BS',
    help='Batch size')

parser.add_argument(
    '--env-gpu',
    type=int,
    default=0,
    help='GPUs to use [-1 CPU only] (default: -1)')

def setup_env_conf (args):
    if args.merger == 'FusionNet':
        merger = merger_FusionNet
    else:
        merger = merger_thres

    if args.spliter == 'FusionNet':
        spliter = spliter_FusionNet
    else:
        spliter = spliter_thres

    env_conf = {
        "corrector_size": [128, 128], 
        "spliter": spliter,
        "merger": merger,
        "cell_thres": int (255 * 0.5),
        "T": args.max_episode_length,
        "agent_out_shape": [1, 2, 2],
        "num_feature": 6,
        "num_action": 2,
        "observation_shape": [256, 256, 6],
        "env_gpu": args.env_gpu
    }
    return env_conf

def get_player(env_conf, viz=False, train=False):
    raw, lbl, prob, gt_lbl = setup_data (env_conf)
    return EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl, obs_format="HWC")

def get_player_test (env_conf):
    raw, lbl, prob, gt_lbl = setup_data (env_conf)
    return lambda: EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl, obs_format="HWC")

class Model(DQNModel):
    def __init__(self, args, env_conf):
        self.args = args
        num_actions = np.prod (env_conf ["agent_out_shape"])
        super(Model, self).__init__(env_conf ["observation_shape"][:2], env_conf ["num_feature"], 1, 
                                        args.algo, num_actions, args.gamma)

    def _get_DQN_prediction(self, image):

        l = (LinearWrap(image)
             # Nature architecture
             # .Conv2D('conv0', 32, 8, strides=4)
             # .Conv2D('conv1', 64, 4, strides=2)
             # .Conv2D('conv2', 64, 3)

             # architecture used for the figure in the README, slower but takes fewer iterations to converge
             .Conv2D('conv0', out_channel=32, kernel_shape=5)
             .MaxPooling('pool0', 2)
             .Conv2D('conv1', out_channel=32, kernel_shape=5)
             .MaxPooling('pool1', 2)
             .Conv2D('conv2', out_channel=64, kernel_shape=4)
             .MaxPooling('pool2', 2)
             .Conv2D('conv3', out_channel=64, kernel_shape=3)

             .FullyConnected('fc0', 512)
             .tf.nn.leaky_relu(alpha=0.01)())

        # DEPTH = 50
        # CFG = {
        #     50: ([3, 4, 6, 3]),
        #     101: ([3, 4, 23, 3]),
        #     152: ([3, 8, 36, 3])
        # }
        # blocks = CFG[DEPTH]

        # image = image
        # image = tf.pad(image, [[0, 0], [3, 2], [3, 2], [0, 0]])
        # image = tf.transpose(image, [0, 3, 1, 2])
        # bottleneck = functools.partial(resnet_bottleneck, stride_first=True)
        # with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
        #               data_format='channels_first'), \
        #         argscope(Conv2D, use_bias=False):
        #     l = (LinearWrap(image)
        #               .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU, padding='VALID')
        #               .MaxPooling('pool0', 3, strides=2, padding='SAME') ())
        #     l = resnet_group ('group0', l, bottleneck, 64, blocks[0], 1)
        #     l = resnet_group ('group1', l, bottleneck, 128, blocks[1], 2)
        #     l = resnet_group ('group2', l, bottleneck, 256, blocks[2], 2)
        #     l = resnet_group ('group3', l, bottleneck, 512, blocks[3], 2)
        #     l = (LinearWrap(l)          
        #               .GlobalAvgPooling('gap')
        #               .FullyConnected('linear', 1000)())

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

def get_config(args, env_conf):
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(env_conf, train=True),
        state_shape=tuple (env_conf ["observation_shape"]),
        batch_size=args.batch_size,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=1
    )

    return AutoResumeTrainConfig(
        data=QueueInput(expreplay),
        model=Model(args, env_conf),
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(
                RunOp(DQNModel.update_target_param, verbose=True),
                every_k_steps=TARGET_NET_UPDATE),    # update target network every 10k steps
            expreplay,
            ScheduledHyperParamSetter('learning_rate',
                                      LEARNING_RATE),
            ScheduledHyperParamSetter(
                ObjAttrParam(expreplay, 'exploration'),
                EXPLORATION,   # 1->0.1 in the first million steps
                interp='linear'),

            PeriodicTrigger(LogVisualizeEpisode(
                ['state'], ['Qvalue'], get_player_test (env_conf)),
                every_k_epochs=args.log_period),

            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=800,
    )


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = str (args.gpu_ids [0])

    env_conf = setup_env_conf (args)
    logger.info("ENV: {}, Num Actions: {}".format(args.env, np.prod (env_conf ["agent_out_shape"])))

    logger.set_logger_dir(
        os.path.join('train_log', args.algo + '-DQN-{}'.format(
            os.path.basename(args.env).split('.')[0])))

    config = get_config(args, env_conf)
    
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(config, SimpleTrainer())
