from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import read_config
from model import *
from train import train
from test import test
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='EM_env',
    metavar='ENV',
    help='environment to train on (default: EM_env)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

parser.add_argument(
    '--workers',
    type=int,
    default=4,
    metavar='W',
    help='how many training processes to use (default: 32)')

parser.add_argument(
    '--num-steps',
    type=int,
    default=4,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=1,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')

parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')

parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')

parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')

parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--save-period',
    type=int,
    default=50,
    metavar='SP',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--log-period',
    type=int,
    default=10,
    metavar='LP',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--shared-optimizer',
    default=True,
    metavar='SO',
    help='use an optimizer without shared statistics.')

parser.add_argument (
    '--hidden-feat',
    type=int,
    default=512,
    metavar='HF')

def setup_env_conf (args):
    env_conf = {
        "agent_out_shape": [1, 8, 8],
        "num_feature": 1,
        "num_action": 1 * 8 * 8,
        "observation_shape": [1, 256, 256],
        "local_wd_size": [32, 32],
    }
    return env_conf

def get_data (path, args):
    base_path = '/home/Pearl/tuan/_Data/ml16-master/segmentation/data/'
    raw_path = base_path + 'train/images/'
    label_path = base_path + 'train/labels/'

    raw_files = natsorted (glob.glob (raw_path + '*.png'))
    lbl_files = natsorted (glob.glob (label_path + '*.png'))

    raw_list = read_im (raw_files)
    lbl_list = read_im (lbl_files)
    i = 0
    while i < len (raw_list):
        if np.sum (lbl_list[i]) == 0:
            del lbl_list [i]
            del raw_list [i]
        else:
            i += 1

    for i in range (len (raw_list)):
        raw_list[i] = np.squeeze (raw_list[i][:,:,0])

    return raw_list, lbl_list

def setup_data (env_conf):
    raw, gt_lbl = get_data (path='Data/train/', args=None)
    return raw, gt_lbl

if __name__ == '__main__':
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')
    env_conf = setup_env_conf (args)
    raw, gt_lbl = setup_data (env_conf)

    # env =EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)
    # shared_model = A3Clstm (env_conf ["observation_shape"], env_conf["num_action"], args.hidden_feat)
    shared_model = SimpleCNN (env_conf ["observation_shape"], env_conf["num_action"])

    if args.load:
        saved_state = torch.load(
            '{0}{1}.dat'.format(args.load_model_dir, args.env),
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()
    
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []

    # p = mp.Process(target=test, args=(args, shared_model, env_conf, [raw, gt_lbl]))
    # p.start()
    # processes.append(p)
    # time.sleep(1)

    for rank in range(0, args.workers):
        p = mp.Process(
            target=train, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl]))
        p.start()
        processes.append(p)
        time.sleep(1)

    for p in processes:
        time.sleep(0.1)
        p.join()

