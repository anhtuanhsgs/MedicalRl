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

from DQNModel import Model as DQNModel
from common import Evaluator, eval_model_multithread, play_n_episodes, LogVisualizeEpisode
from atari_wrapper import FrameStack, MapState, FireResetEnv, LimitLength
from expreplay import ExpReplay
# from atari import AtariPlayer
from refineEnv import *

BATCH_SIZE = 16
# IMAGE_SIZE = (84, 84)
# IMAGE_CHANNEL = None  # 3 in gym and 1 in our own wrapper
# FRAME_HISTORY = 4
# ACTION_REPEAT = 4   # aka FRAME_SKIP

IMAGE_SIZE = (128, 128)
# IMAGE_CHANNEL = None  # 3 in gym and 1 in our own wrapper
IMAGE_CHANNEL = 4
FRAME_HISTORY = 1
ACTION_REPEAT = 1   # aka FRAME_SKIP

UPDATE_FREQ = 4
NUM_ACTIONS = 7

GAMMA = 0.97

# MEMORY_SIZE = 1e6
MEMORY_SIZE = 1e5
# will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
INIT_MEMORY_SIZE = 50000 #1e6 / 20 #MEMORY_SIZE / 20
STEPS_PER_EPOCH = 20000   #100000 // UPDATE_FREQ  # each epoch is 100k played frames
TARGET_NET_UPDATE = 2000 #10000 // UPDATE_FREQ # update target network every 10k steps
EVAL_EPISODE = 1
EPISODE_LOG_PERIOD = 1

USE_GYM = False
ENV_NAME = None
METHOD = None

LEARNING_RATE = [(60, 4e-4), (100, 2e-4), (500, 5e-5)] #[(0, 1e-3), (30, 1e-4), (80, 5e-5), (600, 1e-5)]
EXPLORATION = (0, 1), (10, 0.1), (320, 0.01) # [(0, 1), (50, 0.5), (150, 0.05)]#

raw_list = []
lbl_list = []


def resize_keepdims(im, size):
    # Opencv's resize remove the extra dimension for grayscale images.
    # We add it back.
    ret = cv2.resize(im, size)
    if im.ndim == 3 and ret.ndim == 2:
        ret = ret[:, :, np.newaxis]
    return ret

def init_data ():
    base_path = 'DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)
    for i in range (len (y_train)):
        for img_id in range (len (y_train[i])):
            y_train[i][img_id] = label (y_train[i][img_id] > 0)
        y_train[i] =  y_train[i].astype (np.uint8)

    return X_train [0], y_train[0]

def get_player(viz=False, train=False):
    global raw_list, lbl_list
    if (len (raw_list) == 0):
        raw_list, lbl_list = init_data ()
    SEG_checkpoints_paths = [
        'FCN/checkpoints/128_4/checkpoint_1113000.pth.tar',
        'FCN/checkpoints/128_2/checkpoint_550250.pth.tar',
        'FCN/checkpoints/128_1/checkpoint_496500.pth.tar'
    ]

    return Environment (raw_list, lbl_list, SEG_checkpoints_paths)


class Model(DQNModel):
    def __init__(self):
        super(Model, self).__init__(IMAGE_SIZE, IMAGE_CHANNEL, FRAME_HISTORY, METHOD, NUM_ACTIONS, GAMMA)

    def _get_DQN_prediction(self, image):
        image = image / 255.0
        with argscope(Conv2D, activation=lambda x: PReLU('prelu', x), use_bias=True):
            l = (LinearWrap(image)
                 # architecture used for the figure in the README, slower but takes fewer iterations to converge
                 .Conv2D('conv0', out_channel=32, kernel_shape=9)
                 .MaxPooling('pool0', 2)
                 .Conv2D('conv1', out_channel=32, kernel_shape=5)
                 .MaxPooling('pool1', 2)
                 .Conv2D('conv2', out_channel=64, kernel_shape=4)
                 .MaxPooling('pool2', 2)
                 .Conv2D('conv3', out_channel=64, kernel_shape=3)

                 .FullyConnected('fc0', 512)
                 .tf.nn.leaky_relu(alpha=0.01)
                 ()) 
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
                ['state'], ['Qvalue'], get_player),
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

    raw_list, lbl_list = init_data ()

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
