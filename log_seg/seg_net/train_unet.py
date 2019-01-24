import random
import numpy as np
from natsort import natsorted
import os, sys, argparse, glob
import skimage.io as io
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from ZebrDataset import ZebrDataset
from Unet import *
from Logger import Logger
import multiprocessing as mp
from tqdm import tqdm
import time


# torch.multiprocessing.set_start_method('spawn')

LOG_PERIOD = 5
LOG_DIR = 'log_dir/'
CHECKPOINT_SAVE_PATH = 'checkpoints/'
SAVE_PERIOD = 5
INPUT_SHAPE = (1, 64, 64, 64)
IN_CH = 1
OUT_CH = 1
FEATURES = [8, 16, 32, 64]
BATCH_SIZE = 3

def read_im (paths, downsample=1):
    ret = []
    for path in paths:
        ret.append (io.imread (path)[::downsample,::downsample,::downsample])
    return ret

def save_checkpoint (state, path=CHECKPOINT_SAVE_PATH):
    # print ('Checkpoint saved')
    torch.save (state, path)

def train (train_data, n_epoc, loss_func, optimizer, lr_scheduler, i_iter=0):

    logger = Logger (LOG_DIR)

    for i_ipoc in range (n_epoc):
        pbar = tqdm (total=len (train_data), ascii=True)
        # print('ipoc ' + str (i_ipoc), ' len epoch ', str (len (train_data)))
        ipoc_loss = 0
        
        for i_batch, sample in enumerate (train_data):
            if i_batch == len (train_data):
                break
            pbar.update (1)
            raw = torch.tensor (sample['raw'], device=device, dtype=torch.float32) / 255.0
            target = torch.tensor (sample['lbl'], device=device, dtype=torch.float32) / 255.0
            pred = model (raw)
            
            loss = loss_func (pred, target)

            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()

            ipoc_loss += loss.item () / len (train_data)
            lr_scheduler.step ()

            if i_batch == len (train_data) - 1 and i_ipoc % LOG_PERIOD == 0:
                sys.stdout.flush ()
                # print ('\nWriting log')
                info = {'loss': ipoc_loss, 'learning_rate': lr_scheduler.get_lr () [0]}
                for tag, value in info.items ():
                    logger.scalar_summary (tag, value, i_iter)

                raw = np.expand_dims (raw.detach ().cpu ().numpy() [:,0,:,:], -1)
                target = np.expand_dims (target.detach ().cpu ().numpy ()[:,0,:,:], -1)
                pred = np.expand_dims (pred.detach ().cpu ().numpy ()[:,0,:,:], -1)

                # print (raw.shape, target.shape, pred.shape)

                for tag, value in model.named_parameters ():
                    tag = tag.replace ('.', '/')
                    logger.histo_summary (tag, value.data.cpu ().numpy (), i_iter)

                info = {'train_imgs': [raw, target, pred]}
                for tag, vols in info.items ():
                    for i_img in range (len (vols[0])):
                        raw, target, pred = vols[0][i_img], vols[1][i_img], vols[2][i_img]
                        raw = (raw * 255).astype (np.uint8)
                        target = (target * 255).astype (np.uint8)
                        pred = (pred * 255).astype (np.uint8)

                        z_range, y_range, x_range, nchannel = raw.shape
                        z, y, x = z_range // 2, y_range // 2, x_range // 2
                        # print (vol.shape)
                        yx_raw = raw [z,:,:]
                        zx_raw = raw [:,y,:]
                        yx_lbl = target [z,:,:]
                        zx_lbl = target [:,y,:]
                        yx_pre = pred [z,:,:]
                        zx_pre = pred [:,y,:]
                        yx_log_img = np.concatenate ([yx_raw, yx_lbl, yx_pre], 0)
                        zx_log_img = np.concatenate ([zx_raw, zx_lbl, zx_pre], 0)
                        log_img = np.concatenate ([yx_log_img, zx_log_img], 1)
                        log_img = np.expand_dims (np.repeat (log_img, 3, -1), 0)
                        logger.image_summary (tag + '_' + str (i_img), log_img, i_iter)


            i_iter += 1

        pbar.close ()
        time.sleep (1.0)
        pbar.write (s ='ipoc ' +  str (i_ipoc) + ' iter ' + str (i_iter) + ' loss ' + str (ipoc_loss))

        if i_ipoc % SAVE_PERIOD == 0:
            # tqdm.write ('Checkpoint saved')
            save_checkpoint ({
                'i_iter': i_iter,
                'state_dict': model.state_dict (),
                'optimizer': optimizer.state_dict ()
                }, CHECKPOINT_SAVE_PATH + 'checkpoint_' + str (i_iter) + '.pth.tar')
        # pbar ('\nipoc ' +  str (i_ipoc) + ' iter ' + str (i_iter) + ' loss ', str (ipoc_loss))

def get_data ():
    base_path = '../DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path, downsample=1)
    y_train = read_im (train_label_path, downsample=1)

    return X_train [0], y_train[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--load',    help='load model')
    
    args = parser.parse_args()
    checkpoint_path = None

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.load:
        checkpoint_path = args.load

    print ('Using GPU', os.environ['CUDA_VISIBLE_DEVICES'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup dataflow
    X_train, y_train = get_data ()
    zebrafish_data = ZebrDataset ('train', X_train, y_train, size=INPUT_SHAPE, device=device)
    train_data = DataLoader (zebrafish_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Setup model
    model = Unet (IN_CH, FEATURES, OUT_CH).to (device)
    optimizer = optim.Adam (model.parameters (), lr=1e-4)
    loss_func = nn.BCELoss ()
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    i_iter = 0

    # Load checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load  (checkpoint_path)
        model.load_state_dict (checkpoint['state_dict'])
        i_iter = checkpoint['i_iter']
        optimizer.load_state_dict (checkpoint['optimizer'])

    # Train model
    train (train_data, 10000000, loss_func, optimizer, lr_scheduler, i_iter=i_iter)

    