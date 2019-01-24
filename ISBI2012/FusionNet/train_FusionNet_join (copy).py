import os, sys, argparse, glob
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import resize
from skimage.morphology import erosion, binary_erosion
from skimage.filters import sobel

from ISBI2012_Dataset import *
from FusionNet import FusionNet
from img_aug_func import *
from Logger import Logger
import multiprocessing as mp
from tqdm import tqdm
import time


import os

DS_FACTOR = 1
LOG_PERIOD = 1
LOG_DIR = 'log_dir/'
CHECKPOINT_SAVE_PATH = 'checkpoints/'
SAVE_PERIOD = 5
IN_CH = 2
OUT_CH = 1
BATCH_SIZE = 4
ZOOM_FACTOR = 0.6
refineNet = None
DS_FACTORS_LIST = [1.0, 1.0, 1.0, 1.0]
FEATURES = [16, 32, 64, 128]
LOG_DIR_LIST = []
CHECKPOINT_SAVE_PATH_LIST = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_paths = [
    'checkpoints/0_1.0/checkpoint_90375.pth.tar',
    'checkpoints/2_1.0/checkpoint_154254.pth.tar',
    'checkpoints/1_1.0/checkpoint_112048.pth.tar',
    'checkpoints/3_1.0/checkpoint_163372.pth.tar'
]


def get_data (path, downsample=1):
    train_path = natsorted (glob.glob(path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
    else:
        y_train = np.zeros_like (X_train)

    return X_train, y_train

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint (state, path=CHECKPOINT_SAVE_PATH):
    torch.save (state, path)

def eval (test_data, loss_func, hasTarget=False, models=None):
    with torch.no_grad ():
        ret = []
        for sample in test_data:
            for level in range (4):
                model = models [level]
                raw_t = torch.tensor (sample['raw:' + str (level)], device=device, dtype=torch.float32) / 255.0
                if hasTarget:
                    target_t = torch.tensor (sample['lbl:' + str (level)], device=device, dtype=torch.float32) / 255.0
                pred_t = model (raw_t)
                if hasTarget:
                    loss = loss_func (pred_t, target_t)
                else:
                    loss = 0
                if hasTarget:   
                    ret += [raw_t, target_t, pred_t, loss]
                else:
                    ret += [[raw_t, pred_t, 0]]
            return ret

def visual_log (raw_t, target_t, pred_t, logger, i_iter, log_set, hasTarget=True):
    raw = np.expand_dims (raw_t.detach ().cpu ().numpy()[:,0,:,:], 3)
    mask = np.expand_dims (raw_t.detach ().cpu ().numpy()[:,1,:,:], 3)
    if hasTarget:
        target = np.expand_dims (target_t.detach ().cpu ().numpy ()[:,0,:,:], 3)
    pred = np.expand_dims (pred_t.detach ().cpu ().numpy ()[:,0,:,:], 3)

    if hasTarget:
        concated_imgs = (np.concatenate ([raw, mask, target, pred], 2) * 255).astype (np.uint8)
    else:
        concated_imgs = (np.concatenate ([raw, mask, pred], 2) * 255).astype (np.uint8)

    concated_imgs = np.repeat (concated_imgs, 3, axis=3)

    for tag, value in model.named_parameters ():
        tag = tag.replace ('.', '/')
        logger.histo_summary (tag, value.data.cpu ().numpy (), i_iter)

    info = {log_set + '_imgs': concated_imgs}
    for tag, vol in info.items ():
        for i_img in range (len (vol)):
            logger.image_summary (tag + '_' + str (i_img) + '_' + log_set, vol[i_img:i_img+1], i_iter)

def train_refine (train_data, test_data, n_epoc, loss_func, optimizers, lr_scheduler, i_iter=0):
    if args.refine:
        loggers = []
        print ("log dirs:")
        for level in range (4):
            loggers += [Logger (LOG_DIR_LIST [level])]
            print (LOG_DIR_LIST [level])
    
    for i_ipoc in range (n_epoc):
        pbar = tqdm (total=len (train_data), ascii=True)
        ipoc_loss = 0
        
        for i_batch, sample in enumerate (train_data):
            if i_batch == len (train_data):
                break
            pbar.update (1)
            for level in range (4):
                model = models [level]
                raw_t = torch.tensor (sample['raw:' + str (level)], device=device, dtype=torch.float32) / 255.0
                target_t = torch.tensor (sample['lbl:' + str (level)], device=device, dtype=torch.float32) / 255.0
                
                pred_t = model (raw_t)

                loss = loss_func (pred_t, target_t)

                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

                ipoc_loss += loss.item () / len (train_data)
                lr_scheduler.step ()
                i_iter += 1

                if i_batch == len (train_data) - 1 and (i_ipoc + 1) % LOG_PERIOD == 0:
                    # print ('\nWriting log')
                    info = {'loss': ipoc_loss, 'learning_rate': lr_scheduler.get_lr () [0]}
                    for tag, value in info.items ():
                        loggers [level].scalar_summary (tag, value, i_iter)
                    visual_log (raw_t, target_t, pred_t, loggers[level], i_iter, 'train')
            
            if i_batch == len (train_data) - 1 and (i_ipoc + 1) % LOG_PERIOD == 0:
                raw_pred_loss_list = eval (test_data, loss_func, models=models)
                for level in range (4):
                    raw_t = raw_pred_loss_list [level][0]
                    pred_t = raw_pred_loss_list [level][1]
                    visual_log (raw_t, None, pred_t, loggers[level], i_iter, 'test', hasTarget=False)
                    # info = {'test loss': loss}
                    # for tag, value in info.items ():
                    #     logger.scalar_summary (tag, value, i_iter)
        
        if i_ipoc % SAVE_PERIOD == 0:
            for level in range (4):
                model = models [level]
                save_checkpoint ({
                    'i_iter': i_iter,
                    'state_dict': model.state_dict (),
                    'optimizer': optimizer.state_dict ()
                    }, CHECKPOINT_SAVE_PATH_LIST [level] + 'checkpoint_' + str (i_iter) + '.pth.tar')
        pbar.close ()
        time.sleep (1.0)
        pbar.write (s ='ipoc ' +  str (i_ipoc) + ' iter ' + str (i_iter) + ' loss ' + str (ipoc_loss))

def setup_global_var (args):
    global LOG_DIR, CHECKPOINT_SAVE_PATH, ZOOM_FACTOR, BATCH_SIZE, LOG_DIR_LIST, CHECKPOINT_SAVE_PATH_LIST
    if str.isdigit(args.level):
        level = int (args.level)
    ZOOM_FACTOR = float (args.zoom_factor)
    refine_tag = ''

    if args.refine is not None:
        refine_tag += '_r'
        for level in range (4):
            LOG_DIR_LIST += [LOG_DIR + str (level) + '_' + refine_tag + '/']
            CHECKPOINT_SAVE_PATH_LIST += [CHECKPOINT_SAVE_PATH + str (level) + '_' + refine_tag + '/']
            create_dir (CHECKPOINT_SAVE_PATH_LIST [-1])
    else:
        LOG_DIR = LOG_DIR + args.level + '_' + refine_tag + '/'
        CHECKPOINT_SAVE_PATH = CHECKPOINT_SAVE_PATH + args.level + '_' + refine_tag + '/'
    
        print ('CHECKPOINT_SAVE_PATH:', CHECKPOINT_SAVE_PATH)
        print ('LOG_DIR path', LOG_DIR)
        create_dir (CHECKPOINT_SAVE_PATH)
    
    BATCH_SIZE = int (args.batch_size)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # GPU setup
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print ('Using GPU', os.environ['CUDA_VISIBLE_DEVICES'])  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--gpu',    help='comma seperated list of GPU(s) to use.')
    parser.add_argument ('--level', help='level of model', required=False, default='_')
    parser.add_argument ('--zoom_factor', help='zoom-in factor across level', required=False, default=0.6)
    parser.add_argument ('--load',    help='load model')
    parser.add_argument ('--batch_size', default='4')
    parser.add_argument ('--refine', default=None)
    
    args = parser.parse_args()
    setup_global_var (args)

    checkpoint_path = None
    if args.load:
        checkpoint_path = args.load
    if str.isdigit(args.level):
        level = int (args.level)

    # Setup dataflow
    X_train, y_train = get_data (path='../DATA/train/')
    X_test, y_test = get_data (path='../DATA/test/')

    print ("Train-set size: ", len (X_train))
    print ("Test-set size: ", len (X_test)) 

    gpu_id = int (args.gpu)

    if args.refine is not None:
        models = []
        optimizers = []
        lr_schedulers = []
        for level in range (4):
            model = FusionNet (IN_CH, FEATURES, OUT_CH).to (device)
            model.share_memory ()
            checkpoint = torch.load  (models_paths [level])
            model.load_state_dict (checkpoint['state_dict'])
            models.append (model)

            optimizer = optim.Adam (model.parameters (), lr=1e-4)
            lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)

            optimizers.append (optimizer)
            lr_schedulers.append (optimizer)

        print ("Use refine-dataset")
        isbi2012 = ISBI2012_Dataset16_refine ('test', X_train, y_train, 4, ZOOM_FACTOR, models, [gpu_id, gpu_id, gpu_id, gpu_id])
        isbi2012_test = ISBI2012_Dataset16_refine ('test', X_test, y_test, 4, ZOOM_FACTOR, models, [gpu_id, gpu_id, gpu_id, gpu_id])

    train_data = DataLoader (isbi2012, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_data = DataLoader (isbi2012_test, batch_size=4, num_workers=0)

    # Setup model
    
    optimizers = optim.Adam (model.parameters (), lr=1e-4)
    loss_func = nn.BCELoss ()
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    i_iter = 0

    # Load checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load  (checkpoint_path)
        model.load_state_dict (checkpoint['state_dict'])
        if args.refine is None:
            i_iter = checkpoint['i_iter']
            optimizer.load_state_dict (checkpoint['optimizer'])

    if args.refine is not None:
        train_refine (train_data, test_data, 10000000, loss_func, optimizer, lr_scheduler, i_iter=i_iter)

