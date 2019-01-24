import os, sys, glob, time
import torch
import torch.nn as nn
import torch.optim as optim

from skimage import io
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
from skimage.morphology import erosion, binary_erosion
from skimage.filters import sobel

from setproctitle import setproctitle as ptitle

from ISBI2012_Dataset import ISBI2012_Dataset16_refine
from Logger import Logger
from natsort import natsorted
from img_aug_func import *

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
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

def prepare_dataset (zoom_factor, models, args, level):
    X_train, y_train = get_data (path='../DATA/train/')
    X_test, y_test = get_data (path='../DATA/test/')
    isbi2012 = ISBI2012_Dataset16_refine ('test', X_train, y_train, level + 1, zoom_factor, models, args.gpu_id)
    isbi2012_test = ISBI2012_Dataset16_refine ('test', X_test, y_test, level + 1, zoom_factor, models, args.gpu_id)
    train_data = DataLoader (isbi2012, batch_size=args.batch_size [level], num_workers=0)
    test_data = DataLoader (isbi2012_test, batch_size=4, num_workers=0)
    return train_data, test_data

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

def eval (test_data, loss_func, level, model=None, gpu_id=0, hasTarget=False):
    with torch.no_grad ():
        with torch.cuda.device (gpu_id):
            for sample in test_data:
                raw_t = torch.tensor (sample['raw:' + str (level)], dtype=torch.float32).cuda () / 255.0
                if hasTarget:
                    target_t = torch.tensor (sample['lbl:' + str (level)], dtype=torch.float32).cuda () / 255.0
                pred_t = model (raw_t)
                if hasTarget:
                    loss = loss_func (pred_t, target_t)
                else:
                    loss = 0
                if hasTarget:   
                    return raw_t, target_t, pred_t, loss
                else:
                    return raw_t, pred_t

def train (models, args, level):
    ptitle('Join refine training: {}'.format(level))

    model = models [level]
    optimizer = optim.Adam (model.parameters (), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    gpu_id = args.gpu_id [level % len (args.gpu_id)]
    batch_size = args.batch_size
    zoom_factor = args.zoom_factor
    
    print ('Join refine training: {} \t GPU-id:'.format(level, ))

    optimizers = optim.Adam (model.parameters (), lr=1e-4)
    loss_func = nn.BCELoss ()
    lr_scheduler = optim.lr_scheduler.StepLR (optimizer, step_size=100, gamma=0.999)
    print ('Prepare dataset ...')
    train_data, test_data = prepare_dataset (zoom_factor, models, args, level)
    print ('Finish preparing dataset, start training')
    logger = Logger ('log_dir/' + 'joinrefine_' + str (level) + '/')
    save_path = 'checkpoints/' + 'joinrefine_' + str (level) + '/'

    create_dir (save_path)

    LOG_PERIOD = 5
    SAVE_PERIOD = 5

    i_iter = 0
    for i_ipoc in range (10000000):
        ipoc_loss = 0
        for i_batch, sample in enumerate (train_data):
            if i_batch == len (train_data):
                break
            with torch.cuda.device (gpu_id):
                raw_t = torch.tensor (sample['raw:' + str (level)], dtype=torch.float32).cuda () / 255.0
                target_t = torch.tensor (sample['lbl:' + str (level)], dtype=torch.float32).cuda () / 255.0
                pred_t = model (raw_t)
                loss = loss_func (pred_t, target_t)

                optimizer.zero_grad ()
                loss.backward ()
                optimizer.step ()

            ipoc_loss += loss.item () / len (train_data)
            lr_scheduler.step ()
            i_iter += 1

        print('level: {}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
            level, i_ipoc, ipoc_loss))

        info = {'loss': ipoc_loss, 'learning_rate': lr_scheduler.get_lr () [0]}
        for tag, value in info.items ():
            logger.scalar_summary (tag, value, i_iter)
        visual_log (raw_t, target_t, pred_t, logger, i_iter, 'train')

        if (i_ipoc + 1) % LOG_PERIOD == 0:
            raw_t, pred_t = eval (test_data, loss_func, model=model, level=level, gpu_id=gpu_id)
            visual_log (raw_t, None, pred_t, logger, i_iter, 'test', hasTarget=False)

        if i_ipoc % SAVE_PERIOD == 0:
            torch.save ({
                'i_iter': i_iter,
                'state_dict': model.state_dict (),
                'optimizer': optimizer.state_dict ()
                }, save_path + 'checkpoint_' + str (i_iter) + '.pth.tar')
