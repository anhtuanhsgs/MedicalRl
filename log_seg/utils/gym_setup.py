'''
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/dqn_utils.py
https://github.com/berkeleydeeprlcourse/homework/blob/master/hw3/run_dqn_atari.py
'''

import gym
from gym import wrappers
import numpy as np
import random
from utils.atari_wrappers import *
from refineEnv import *

SEG_CHECKPOINT_PATH = 'seg_net/checkpoints/checkpoint_34713.pth.tar'

def set_global_seeds(i):
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_env(task, seed, vid_dir_name, double_dqn, dueling_dqn):
    # env_id = task.env_id
    # env_id = 

    env = gym.make('Breakout-v0')
    vid_dir_name = 'Breakout-v0'

    set_global_seeds(seed)
    env.seed(seed)

    if double_dqn:
        expt_dir = 'tmp/%s/double/' %vid_dir_name
    elif dueling_dqn:
        expt_dir = 'tmp/%s/dueling/' %vid_dir_name
    else:
        expt_dir = 'tmp/%s/' %vid_dir_name
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env

def init_data ():
    base_path = 'data/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    return X_train [0], y_train[0]

raw_list = []
lbl_list = []

def down_sample_3d (data_list, factor):
    ret = []
    for data in data_list:
        assert (len (data.shape) == 3)
        ret += [data[::factor, ::factor, ::factor]]
    return ret

def get_data ():
    global raw_list, lbl_list
    if (len (raw_list) != 0):
        return raw_list, lbl_list
    base_path = 'DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    X_train = X_train [0] / 65535.0 * 255
    y_train = y_train [0]

    return X_train, y_train


def get_medical_env ():
    global raw_list, lbl_list
    if (len (raw_list) == 0):
        raw_list, lbl_list = get_data ()
        lbl_list = label (erode_label ([lbl_list > 0]) [0])
        raw_list.flags.writeable = False
        lbl_list.flags.writeable = False
    return Environment (raw_list, lbl_list, SEG_CHECKPOINT_PATH)

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)
