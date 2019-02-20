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
from CorrectorModule.corrector_utils import *
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
    default=2,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=6,
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

parser.add_argument (
    '--use-stop', 
    action='store_true'
)

# parser.add_argument(
#     '--env-gpu',
#     type=int,
#     default=0,
#     help='GPUs to use [-1 CPU only] (default: -1)')

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
    default=96,
    metavar='TLP',
)

parser.add_argument (
    '--spliter',
    default='FusionNet',
    metavar='SPL',
    choices=['FusionNet', 'Thres', 'UNet'])

parser.add_argument (
    '--merger',
    default='FusionNet',
    metavar='MER',
    choices=['FusionNet', 'Thres', 'UNet'])

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)

parser.add_argument (
    '--hidden-feat',
    type=int,
    default=512,
    metavar='HF')

parser.add_argument (
    '--reward-thres',
    type=float,
    default=0.4,
    metavar='RT'
)


parser.add_argument(
    '--merge-err',
    action='store_true'
)

parser.add_argument(
    '--split-err',
    action='store_true'
)

parser.add_argument (
    '--alpha',
    type=float,
    default=4,
    metavar='RT'
)

parser.add_argument (
    '--beta',
    type=float,
    default=2,
    metavar='RT'
)

parser.add_argument (
    '--impr-fusion',
    type=str,
    default=None,
   
    metavar='if'
)

parser.add_argument (
    '--impr-unet',
    type=str,
    default=None,
    metavar='if'
)

parser.add_argument (
    '--prob-path',
    type=str,
    default='Data/train/deploy/normal.tif'
)

parser.add_argument (
    '--prob-path-test',
    type=str,
    default='Data/test/deploy/normal.tif'
)

parser.add_argument (
    '--gauss-blending',
    action='store_true'
)

parser.add_argument (
    '--continuous',
    action='store_true'
)

parser.add_argument (
    '--num-err',
    type=int,
    default=6
)

parser.add_argument (
    '--multires',
    action='store_true',
)

def setup_env_conf (args):
    if "EM_env" in args.env:
        if args.merger == 'FusionNet':
            merger = merger_FusionNet_fn
        elif args.merger == "Thres":
            merger = merger_thres_fn
        elif args.merger == "UNet":
            merger = unet_160_fn
    else:
        merger = merger_thres_fn

    if "EM_env" in args.env:
        if args.spliter == 'FusionNet':
            spliter = spliter_FusionNet_fn
        elif args.spliter == "Thres":
            spliter = spliter_thres_fn
        elif args.spliter == "UNet":
            spliter = unet_96_fn
    else:
        #############Single case error###############
        if args.merge_err:
            spliter = spliter_thres_fn
        else:
            spliter = merger_thres_fn

    if args.multires:
        spliter = unet_96_fn
        merger = unet_160_fn
        args.env += '_multires'

    if args.multires:
        corrector_size = [[96, 96], [160, 160]]
    else:
        corrector_size = [96, 96]

    env_conf = {
        "corrector_size": corrector_size, 
        "spliter": spliter,
        "merger": merger,
        "cell_thres": int (255 * 0.35),
        "T": args.max_episode_length,
        "agent_out_shape": [1, 5, 5],
        "observation_shape": [3, 256, 256],
        "reward_thres": args.reward_thres,
        "num_segs": 40,
        "split_err": args.split_err,
        "merge_err": args.merge_err,
        "alpha": args.alpha,
        "beta": args.beta,
        "prob_path": args.prob_path,
        "prob_path_test": args.prob_path_test,
        "gauss-blending": args.gauss_blending,
        "continuous": args.continuous,
        "use_stop": args.use_stop,
        "num_err": args.num_err,
        "multires": args.multires
    }

    if args.multires:
        env_conf ["agent_out_shape"][0] = 2

    if (args.env != "EM_env"):
        if args.split_err:
            args.env += "_split"
        if args.merge_err:
            args.env += "_merge"

    if args.continuous:
        args.env += '_cont'

    if args.impr_fusion is not None:
        print ("Train improve-FusionNet ")
        args.env += "_impr_fusion_" + args.impr_fusion

    if args.impr_unet is not None:
        print ("Train improve-unet ")
        args.env += "_impr_unet_" + args.impr_unet

    args.log_dir += args.env + "/"

    env_conf ["num_action"] = int (np.prod (env_conf ['agent_out_shape']))
    if env_conf ["use_stop"]:
        env_conf ["num_action"] += 1
    if args.continuous:
        env_conf ["num_action"] = 2
    env_conf ["num_feature"] = env_conf ['observation_shape'][0]
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
    prob = io.imread (env_conf["prob_path"])
    prob = np.clip (prob, int (255 * 0.1), int (255 * 0.9))
    ##################################
    # prob = np.zeros_like (prob)
    ##################################
    lbl = []
    for img in prob:
        lbl += [label (img > env_conf ['cell_thres'])]
    lbl = np.array (lbl)

    return raw, lbl, prob, gt_lbl

def setup_data_test (env_conf):
    raw, gt_lbl = get_data (path='Data/test/', args=None)
    prob = io.imread (env_conf["prob_path_test"])
    prob = np.clip (prob, int (255 * 0.1), int (255 * 0.9))
    lbl = []
    for img in prob:
        lbl += [label (img > env_conf ['cell_thres'])]
    lbl = np.array (lbl)
    gt_lbl = np.zeros (lbl.shape, dtype=np.uint8) + 127

    return raw, lbl, prob, gt_lbl

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
        raw, lbl, prob, gt_lbl = setup_data (env_conf)
        raw_test, lbl_test, prob_test, gt_lbl_test = setup_data_test (env_conf)

    # env =EM_env (raw, lbl, prob, env_conf, 'train', gt_lbl)
    if not args.continuous:
        shared_model = A3Clstm (env_conf ["observation_shape"], 
                            env_conf["num_action"], args.hidden_feat)
    else:
        shared_model = A3Clstm_continuous (env_conf ["observation_shape"], 
                            env_conf["num_action"], args.hidden_feat)


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
    # if "EM_env" in args.env:
    #     p = mp.Process(target=test, args=(args, shared_model, env_conf, [raw, lbl, prob, gt_lbl], True))
    # else:
    #     p = mp.Process(target=test, args=(args, shared_model, env_conf))
    # p.start()
    # processes.append(p)
    # time.sleep(1)

    # if "EM_env" in args.env:
    #     p = mp.Process(target=test, args=(args, shared_model, env_conf, 
    #         [raw_test, lbl_test, prob_test, gt_lbl_test], False))
    #     p.start()
    #     processes.append(p)
    #     time.sleep(1)

    for rank in range(0, args.workers):
        if "EM_env" in args.env:
            p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf, [raw, lbl, prob, gt_lbl]))
        else:
             p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(1)

    for p in processes:
        time.sleep(0.1)
        p.join()

