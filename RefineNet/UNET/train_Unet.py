import os, sys, argparse, glob
from natsort import natsorted

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import resize

from CremiDataset import CremiDataset
from Unet import Unet
from img_aug_func import *
from Logger import Logger
import multiprocessing as mp
from tqdm import tqdm
import time

import os

LOG_PERIOD = 1
LOG_DIR = 'log_dir/'
CHECKPOINT_SAVE_PATH = 'checkpoints/'
SAVE_PERIOD = 5
IN_CH = 1
OUT_CH = 1
FEATURES = [8, 16, 32, 64]
BATCH_SIZE = 8

def get_data (downsample=1, ):
    base_path = '../DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    return X_train [0], y_train[0]

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint (state, path=CHECKPOINT_SAVE_PATH):
    torch.save (state, path)

def eval (test_data, loss_func):
    for sample in test_data:
        raw_t = torch.tensor (sample['raw'], device=device, dtype=torch.float32) / 255.0
        target_t = torch.tensor (sample['lbl'], device=device, dtype=torch.float32) / 255.0
        pred_t = model (raw_t)
        loss = loss_func (pred_t, target_t)
        return raw_t, target_t, pred_t, loss

def visual_log (raw_t, target_t, pred_t, logger, i_iter, log_set):
    raw = np.expand_dims (raw_t.detach ().cpu ().numpy()[:,0,:,:], 3)
    mask = np.expand_dims (raw_t.detach ().cpu ().numpy()[:,1,:,:], 3)
    target = np.expand_dims (target_t.detach ().cpu ().numpy ()[:,0,:,:], 3)
    pred = np.expand_dims (pred_t.detach ().cpu ().numpy ()[:,0,:,:], 3)

    concated_imgs = (np.concatenate ([raw, mask, target, pred], 2) * 255).astype (np.uint8)
    concated_imgs = np.repeat (concated_imgs, 3, axis=3)

    for tag, value in model.named_parameters ():
        tag = tag.replace ('.', '/')
        logger.histo_summary (tag, value.data.cpu ().numpy (), i_iter)

    info = {'train_imgs': concated_imgs}
    for tag, vol in info.items ():
        for i_img in range (len (vol)):
            logger.image_summary (tag + '_' + str (i_img) + '_' + log_set, vol[i_img:i_img+1], i_iter)

def train (train_data, test_data, n_epoc, loss_func, optimizer, lr_scheduler, i_iter=0):
    logger = Logger (LOG_DIR)
    
    for i_ipoc in range (n_epoc):
        pbar = tqdm (total=len (train_data), ascii=True)
        ipoc_loss = 0
        
        for i_batch, sample in enumerate (train_data):
            if i_batch == len (train_data):
                break
            pbar.update (1)
            raw_t = torch.tensor (sample['raw'], device=device, dtype=torch.float32) / 255.0
            target_t = torch.tensor (sample['lbl'], device=device, dtype=torch.float32) / 255.0

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
                    logger.scalar_summary (tag, value, i_iter)
                visual_log (raw_t, target_t, pred_t, logger, i_iter, 'train')
                raw_t, target_t, pred_t, loss = eval (test_data, loss_func)
                visual_log (raw_t, target_t, pred_t, logger, i_iter, 'test')
                info = {'test loss': loss}
                for tag, value in info.items ():
                    logger.scalar_summary (tag, value, i_iter)
        
        if i_ipoc % SAVE_PERIOD == 0:
            
            save_checkpoint ({
                'i_iter': i_iter,
                'state_dict': model.state_dict (),
                'optimizer': optimizer.state_dict ()
                }, CHECKPOINT_SAVE_PATH + 'checkpoint_' + str (i_iter) + '.pth.tar')
        pbar.close ()
        time.sleep (1.0)
        pbar.write (s ='ipoc ' +  str (i_ipoc) + ' iter ' + str (i_iter) + ' loss ' + str (ipoc_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--gpu',    help='comma seperated list of GPU(s) to use.')
    parser.add_argument ('--load',   help='load model')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parser.parse_args()
    # GPU setup
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_dir (CHECKPOINT_SAVE_PATH)

    X_data, y_data = get_data ()
    X_train, y_train = X_data [5:], y_data [5:]
    X_test, y_test = X_data [:5], y_data[:5]

    print ("Train-set size: ", len (X_train))
    print ("Test-set size: ", len (X_test)) 

    cremiData = CremiDataset (X_train, y_train)
    cremiData_test = CremiDataset (X_test, y_test)

    train_data = DataLoader (cremiData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_data = DataLoader (cremiData_test, batch_size=4, num_workers=0)
    model = Unet (IN_CH, FEATURES, OUT_CH).to (device)
    optimizer = optim.Adam (model.parameters (), lr=1e-4)
    loss_func = nn.BCELoss ()
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    i_iter = 0

     # Train model
    train (train_data, test_data, 10000000, loss_func, optimizer, lr_scheduler, i_iter=i_iter)