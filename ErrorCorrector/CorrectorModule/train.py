import os, sys, glob, time
from os import sys, path
import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import sobel

from EMDataset import EMDataset

from natsort import natsorted
sys.path.append('../')
from Utils.img_aug_func import *
from Utils.utils import *
from Utils.Logger import Logger
import matplotlib.pyplot as plt
import Losses

LOG_PERIOD = 5
SAVE_PERIOD = 5

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
        y_train = get_cell_prob (y_train, dilation=args.dilation, erosion=args.erosion)
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train


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

    info = {log_set + '_imgs': concated_imgs}
    for tag, vol in info.items ():
        for i_img in range (len (vol)):
            logger.image_summary (tag + '_' + str (i_img) + '_' + log_set, vol[i_img:i_img+1], i_iter)


def eval (test_data, loss_func, model=None, gpu_id=0, hasTarget=False):
    with torch.no_grad ():
        with torch.cuda.device (gpu_id):
            for sample in test_data:
                raw_t = torch.tensor (sample['raw'], dtype=torch.float32).cuda () / 255.0
                if hasTarget:
                    target_t = torch.tensor (sample['lbl'], dtype=torch.float32).cuda () / 255.0
                pred_t = model (raw_t)
                if hasTarget:
                    loss = loss_func (pred_t, target_t)
                else:
                    loss = 0
                if hasTarget:   
                    return raw_t, target_t, pred_t, loss
                else:
                    return raw_t, pred_t

def train (model, args, name):

    model = model
    optimizer = optim.Adam (model.parameters (), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    gpu_id = args.gpu_id 
    batch_size = args.batch_size
    
    optimizers = optim.Adam (model.parameters (), lr=1e-4)

    if args.loss == 'WBCE':
        loss_func = Losses.weighted_binary_cross_entropy
    else:
        loss_func = nn.BCELoss ()
    
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    print ('Prepare dataset ...')
    train_data, test_data = prepare_dataset (model, args)
    print ('Finish preparing dataset, start training')
    logger = Logger ('log_dir/' + name + '/')
    save_path = 'checkpoints/' + name + '/'

    create_dir (save_path)

    i_iter = 0
    for i_ipoc in range (10000000):
        ipoc_loss = 0
        for i_batch, sample in enumerate (train_data):
            if i_batch == len (train_data):
                break
            with torch.cuda.device (gpu_id):
                raw_t = torch.tensor (sample['raw'], dtype=torch.float32).cuda () / 255.0
                target_t = torch.tensor (sample['lbl'], dtype=torch.float32).cuda () / 255.0
                pred_t = model (raw_t)
                if args.loss == 'weighted':
                    if args.weights is not None:
                        weights = args.weights
                    else:
                        ESP = 1e-5
                        neg_weight = torch.sum (target_t) / (1.0 * np.prod (target_t.shape)) + ESP
                        weights = [neg_weight, 1 - neg_weight]
                    loss = loss_func (pred_t, target_t, weights)    
                else:
                    loss = loss_func (pred_t, target_t)

                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

            ipoc_loss += loss.item () / len (train_data)
            lr_scheduler.step ()
            i_iter += 1

        print('type: {}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
            args.type, i_ipoc, ipoc_loss))

        info = {'loss': ipoc_loss, 'learning_rate': lr_scheduler.get_lr () [0]}
        for tag, value in info.items ():
            logger.scalar_summary (tag, value, i_iter)
        visual_log (raw_t, target_t, pred_t, logger, i_iter, 'train')

        if (i_ipoc + 1) % LOG_PERIOD == 0:
            raw_t, pred_t = eval (test_data, loss_func, hasTarget=False, model=model, gpu_id=gpu_id)
            visual_log (raw_t, None, pred_t, logger, i_iter, 'test', hasTarget=False)

        if i_ipoc % SAVE_PERIOD == 0:
            torch.save ({
                'i_iter': i_iter,
                'state_dict': model.state_dict (),
                'optimizer': optimizer.state_dict ()
                }, save_path + 'checkpoint_' + str (i_iter) + '.pth.tar')                  