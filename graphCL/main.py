from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import read_config
from models.models import *
from train import train
from test import test
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam
from models.models import *


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='Voronoi_env',
    metavar='ENV',
    help='environment to train on (default: Voronoi_env)')

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
    default=1,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=3,
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
    help='Save period')

parser.add_argument(
    '--log-period',
    type=int,
    default=5,
    metavar='LP',
    help='Log period')

parser.add_argument (
    '--train-log-period',
    type=int,
    default=32,
    metavar='TLP',
)

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)

parser.add_argument (
    '--hidden-feat',
    type=int,
    default=64,
    metavar='HF'
)

parser.add_argument (
    '--radius',
    type=int,
    default=16,
)

parser.add_argument (
    '--speed',
    type=int,
    default=2,
)

parser.add_argument (
    '--features',
    type=int,
    default= [32, 64, 128, 256],
    nargs='+'
)

parser.add_argument (
    '--size',
    type=int,
    default= [96, 96],
    nargs='+'
)

parser.add_argument (
    '--model',
    default='UNet',
    choices=['UNet', 'FusionNetLstm', "FusionNet", "UNetLstm"]
)

parser.add_argument (
    "--reward",
    default="normal",
    choices=["normal", "gaussian", "density"]
)

parser.add_argument (
    "--use-lbl",
    action="store_true"
)

def setup_env_conf (args):

    env_conf = {
        "T": args.max_episode_length,
        "size": args.size,
        "num_segs": 12,
        "radius": args.radius,
        "speed": args.speed,
        "reward": args.reward,
        "use_lbl": args.use_lbl,
    }
    env_conf ["observation_shape"] = [env_conf ["T"] + 1] + env_conf ["size"]

    if "Lstm" in args.model:
        args.env += "_lstm"
    if args.use_lbl:
        args.env += "_lbl"
        env_conf ["observation_shape"][0] = 2
    args.env += "_" + args.reward
    args.log_dir += args.env + "/"

    return env_conf

def get_cell_prob (lbl, dilation, erosion):
    ESP = 1e-5
    elevation_map = []
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    elevation_map = elevation_map > ESP
    cell_prob = ((lbl > 0) ^ elevation_map) & (lbl > 0)
    for i in range (len (cell_prob)):
        for j in range (erosion):
            cell_prob [i] = binary_erosion (cell_prob [i])
    for i in range (len (cell_prob)):
        for j in range (dilation):
            cell_prob [i] = binary_dilation (cell_prob [i])
    return np.array (cell_prob, dtype=np.uint8) * 255

def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
        gt_prob = get_cell_prob (y_train, 0, 0)
        y_train = []
        for img in gt_prob:
            y_train += [label (img).astype (np.int32)]
        y_train = np.array (y_train)
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train
 
def setup_data (env_conf):
    raw , gt_lbl = get_data (path='Data/train/', args=None)
    raw = raw 
    gt_lbl = gt_lbl
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

    if "EM_env" in args.env:
        raw, gt_lbl = setup_data (env_conf)

    if (args.model == 'UNet'):
        shared_model = UNet (env_conf ["observation_shape"][0], args.features, 2)
    elif (args.model == "FusionNetLstm"):
        shared_model = FusionNetLstm (env_conf ["observation_shape"], args.features, 2, args.hidden_feat)
    elif (args.model == "FusionNet"):
        shared_model = FusionNet (env_conf ["observation_shape"][0], args.features, 2)
    elif (args.model == "UNetLstm"):
        shared_model = UNetLstm (env_conf ["observation_shape"], args.features, 2, args.hidden_feat)

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
    if "EM_env" in args.env:
        p = mp.Process(target=test, args=(args, shared_model, env_conf, [raw, gt_lbl], True))
    else:
        p = mp.Process(target=test, args=(args, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    # if "EM_env" in args.env:
    #     p = mp.Process(target=test, args=(args, shared_model, env_conf, 
    #         [raw_test, lbl_test, prob_test, gt_lbl_test], False))
    #     p.start()
    #     processes.append(p)
    #     time.sleep(1)

    for rank in range(0, args.workers):
        if "EM_env" in args.env:
            p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl]))
        else:
             p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

