#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import gym

from tensorpack import *
from resnet_model import *

from DQNModel import Model as DQNModel
from common import Evaluator, eval_model_multithread, play_n_episodes, LogVisualizeEpisode
from atari_wrapper import FrameStack, MapState, FireResetEnv, LimitLength
from expreplay import ExpReplay
# from atari import AtariPlayer
from refineEnv import *
from natsort import natsorted
import os, sys, argparse, glob
import skimage.io as io
from skimage.morphology import erosion, binary_erosion
from skimage.filters import sobel
import functools

BATCH_SIZE = 16
# IMAGE_SIZE = (84, 84)
# IMAGE_CHANNEL = None  # 3 in gym and 1 in our own wrapper
# FRAME_HISTORY = 4
# ACTION_REPEAT = 4   # aka FRAME_SKIP

IMAGE_SIZE = (256, 256)
# IMAGE_CHANNEL = None  # 3 in gym and 1 in our own wrapper
IMAGE_CHANNEL = 4
FRAME_HISTORY = 1
ACTION_REPEAT = 1   # aka FRAME_SKIP

UPDATE_FREQ = 4
NUM_ACTIONS = 7

GAMMA = 0.99

# MEMORY_SIZE = 1e6
MEMORY_SIZE = 1e5
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = 1e6 / 20 #MEMORY_SIZE / 20
STEPS_PER_EPOCH = 100000 // UPDATE_FREQ  # each epoch is 100k played frames
TARGET_NET_UPDATE = 10000 // UPDATE_FREQ # update target network every 10k steps
EVAL_EPISODE = 1
EPISODE_LOG_PERIOD = 1
ZOOM_FACTOR = 0.6

USE_GYM = False
ENV_NAME = None
METHOD = None

LEARNING_RATE = [(60, 4e-4), (100, 2e-4), (500, 5e-5)] #[(0, 1e-3), (30, 1e-4), (80, 5e-5), (600, 1e-5)]
EXPLORATION = (0, 1), (1, 0.9), (10, 0.9), (20, 0.1), (320, 0.01) # [(0, 1), (50, 0.5), (150, 0.05)]#

raw_list = []
lbl_list = []
raw_list_test = []
lbl_list_test = []


def resize_keepdims(im, size):
    # Opencv's resize remove the extra dimension for grayscale images.
    # We add it back.
    ret = cv2.resize(im, size)
    if im.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    return ret

def get_cell_prob (lbl):
    elevation_map = []
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    elevation_map = elevation_map > 0
    cell_prob = (lbl > 0) ^ elevation_map
    for i in range (len (cell_prob)):
        cell_prob [i] = binary_erosion (cell_prob [i])
    return np.array (cell_prob, dtype=np.uint8) * 255

def init_data (path):
    train_path = natsorted (glob.glob(path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'trainB/*.tif'))
    X_train = read_im (train_path) [0]
    y_train = read_im (train_label_path) [0]
    # y_train = get_cell_prob (y_train)
    # for i in range (len (y_train)):
    for img_id in range (len (y_train)):
        y_train[img_id] = label (y_train[img_id] > 0)

    return X_train , y_train

def get_player(viz=False, train=False):
    global raw_list, lbl_list
    if (len (raw_list) == 0):
        raw_list, lbl_list = init_data ('DATA/train/')
    SEG_checkpoints_paths = [
        'checkpoints/0_1.0/checkpoint_226500.pth.tar',
        'checkpoints/1_1.0/checkpoint_451500.pth.tar',
        'checkpoints/2_1.0/checkpoint_706500.pth.tar',
        'checkpoints/3_1.0/checkpoint_759000.pth.tar'
    ]
    for i in range (len (SEG_checkpoints_paths)):
        SEG_checkpoints_paths[i] = 'FusionNet/' + SEG_checkpoints_paths[i]
    print ("train len:", len (raw_list))
    return Environment (raw_list, lbl_list, SEG_checkpoints_paths, env_type='train')

def get_player_test ():
    global raw_list_test, lbl_list_test
    if (len (raw_list_test) == 0):
        raw_list_test, lbl_list_test = init_data ('DATA/train/')
    SEG_checkpoints_paths = [
        'checkpoints/0_1.0/checkpoint_226500.pth.tar',
        'checkpoints/1_1.0/checkpoint_451500.pth.tar',
        'checkpoints/2_1.0/checkpoint_706500.pth.tar',
        'checkpoints/3_1.0/checkpoint_759000.pth.tar'
    ]
    for i in range (len (SEG_checkpoints_paths)):
        SEG_checkpoints_paths[i] = 'FusionNet/' + SEG_checkpoints_paths[i]
    print ("test len:", len (raw_list_test))
    return Environment (raw_list_test, lbl_list_test, SEG_checkpoints_paths, env_type='test')

class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, IMAGE_CHANNEL, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):

        DEPTH = 50
        CFG = {
            50: ([3, 4, 6, 3]),
            101: ([3, 4, 23, 3]),
            152: ([3, 8, 36, 3])
        }
        blocks = CFG[DEPTH]

        image = image / 255.0
        image = tf.pad(image, [[0, 0], [3, 2], [3, 2], [0, 0]])
        image = tf.transpose(image, [0, 3, 1, 2])
        bottleneck = functools.partial(resnet_bottleneck, stride_first=True)
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
                      data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):
            l = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU, padding='VALID')
                      .MaxPooling('pool0', 3, strides=2, padding='SAME') ())
            l = resnet_group ('group0', l, bottleneck, 64, blocks[0], 1)
            l = resnet_group ('group1', l, bottleneck, 128, blocks[1], 2)
            l = resnet_group ('group2', l, bottleneck, 256, blocks[2], 2)
            l = resnet_group ('group3', l, bottleneck, 512, blocks[3], 2)
            l = (LinearWrap(l)          
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1024)())

        if self.method != 'Dueling':
            Q = FullyConnected('fct', l, self.num_actions)
        else:
            # Dueling DQN
            V = FullyConnected('fctV', l, 1)
            As = FullyConnected('fctA', l, self.num_actions)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')

def get_config():
    expreplay = ExpReplay(
        predictor_io_names=(['state'], ['Qvalue']),
        player=get_player(train=True),
        state_shape=IMAGE_SIZE + (IMAGE_CHANNEL,),
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        init_memory_size=INIT_MEMORY_SIZE,
        init_exploration=1.0,
        update_frequency=UPDATE_FREQ,
        history_len=FRAME_HISTORY
    )

    return AutoResumeTrainConfig(
        data=QueueInput(expreplay),
        model=Model(),
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
            # PeriodicTrigger(Evaluator(
            #     EVAL_EPISODE, ['state'], ['Qvalue'], get_player),
            #     every_k_epochs=10),

            PeriodicTrigger(LogVisualizeEpisode(
                ['state'], ['Qvalue'], get_player_test),
                every_k_epochs=EPISODE_LOG_PERIOD),

            HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=800,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
                        choices=['play', 'eval', 'train'], default='train')
    # parser.add_argument('--env', required=True,
                        # help='either an atari rom file (that ends with .bin) or a gym atari environment name')
    parser.add_argument('--algo', help='algorithm',
                        choices=['DQN', 'Double', 'Dueling'], default='Double')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ENV_NAME = args.env
    ENV_NAME = 'Medical_IMG_ENV'
    # USE_GYM = not ENV_NAME.endswith('.bin')
    # IMAGE_CHANNEL = 3 if USE_GYM else 1
    # IMAGE_CHANNEL = 5
    METHOD = args.algo
    # set num_actions
    # NUM_ACTIONS = get_player().action_space.n
    logger.info("ENV: {}, Num Actions: {}".format(ENV_NAME, NUM_ACTIONS))

    raw_list, lbl_list = init_data ('DATA/train/')

    if args.task != 'train':
        assert args.load is not None
        pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader(args.load),
            input_names=['state'],
            output_names=['Qvalue']))
        if args.task == 'play':
            # play_n_episodes(get_player(viz=0.01), pred, 100)
            play_n_episodes(get_player(), pred, 100)
        elif args.task == 'eval':
            eval_model_multithread(pred, EVAL_EPISODE, get_player)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', METHOD + '-DQN-{}'.format(
                os.path.basename(ENV_NAME).split('.')[0])))
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SimpleTrainer())
