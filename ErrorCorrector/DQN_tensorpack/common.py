# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu
import random
import time
import multiprocessing
from tqdm import tqdm
from six.moves import queue
import numpy as np

from tensorpack.utils.concurrency import StoppableThread, ShareSessionThread
from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs

class LogVisualizeEpisode (Callback):

    def __init__ (self, input_names, output_names, get_player_fn):
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph (self):
        self.pred = self.trainer.get_predictor (self.input_names, self.output_names)

    def _trigger (self):
        player = self.get_player_fn ()
        current_obs = player.reset ()
        log_imgs = [player.render ()]

        done = False
        step = 0
        tot_reward = 0
        stack = []

        start_time = time.time ()
        while not done:
            step += 1

            state_action_values = self.pred (np.expand_dims (current_obs, 0)) [0]
            state_action_values = np.squeeze (state_action_values)
            action = np.argmax (state_action_values)            
        
            line_header = time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time))
            
            obs_from_step, reward, done, info = player.step (action)
            print (line_header)
            print ('Step :', step)
            print ('State_action_value:')
            print ('Choose:', player.int2index (action, player.agent_out_shape))
            print ('reward:', reward)

            current_obs = obs_from_step
            log_imgs += [player.render ()]

        concated_img = np.concatenate (log_imgs, 0)
        self.trainer.monitors.put_image ('Log episode', concated_img)
        print ('Final score:', player.old_score)
