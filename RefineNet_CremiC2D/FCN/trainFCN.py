import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from CremiDataset import CremiDataset
from FCN2D import *
from img_aug_func import *
from Logger import Logger
import multiprocessing as mp
from tqdm import tqdm
import time

import os

DOWNSAMPLE_FACTOR = 1
LOG_PERIOD = 5
LOG_DIR = 'log_dir/'
CHECKPOINT_SAVE_PATH = 'checkpoints/'
SAVE_PERIOD = 5
INPUT_SHAPE = (1, 64, 64, 64)
IN_CH = 2
OUT_CH = 1
BATCH_SIZE = 4
refineNet = None


def get_data (downsample=1):
    base_path = 'DATA/'
    train_path = natsorted (glob.glob(base_path + 'trainA/*.tif'))
    train_label_path = natsorted (glob.glob(base_path + 'trainB/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    return X_train [0], y_train[0]

def create_dir (directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint (state, path=CHECKPOINT_SAVE_PATH):
    # print ('Checkpoint saved')
    torch.save (state, path)

def train (train_data, n_epoc, loss_func, optimizer, lr_scheduler, i_iter=0):

    logger = Logger (LOG_DIR)

    for i_ipoc in range (n_epoc):
        pbar = tqdm (total=len (train_data), ascii=True, smoothing=True, leave=True)
        sys.stdout.flush()
        time.sleep (1.0)
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
                time.sleep (1.0)
                # print ('\nWriting log')
                info = {'loss': ipoc_loss, 'learning_rate': lr_scheduler.get_lr () [0]}
                for tag, value in info.items ():
                    logger.scalar_summary (tag, value, i_iter)

                raw = np.expand_dims (raw.detach ().cpu ().numpy()[:,0,:,:], 3)
                mask = np.expand_dims (raw.detach ().cpu ().numpy()[:,1,:,:], 3)
                target = np.expand_dims (target.detach ().cpu ().numpy ()[:,0,:,:], 3)
                pred = np.expand_dims (pred.detach ().cpu ().numpy ()[:,0,:,:], 3)

                # print (raw.shape, target.shape, pred.shape)

                concated_imgs = (np.concatenate ([raw, target, pred], 2) * 255).astype (np.uint8)
                concated_imgs = np.repeat (concated_imgs, 3, axis=3)

                for tag, value in model.named_parameters ():
                    tag = tag.replace ('.', '/')
                    logger.histo_summary (tag, value.data.cpu ().numpy (), i_iter)

                info = {'train_imgs': concated_imgs}
                for tag, vol in info.items ():
                    for i_img in range (len (vol)):
                        logger.image_summary (tag + '_' + str (i_img), vol[i_img:i_img+1], i_iter)

            i_iter += 1
        
        if i_ipoc % SAVE_PERIOD == 0:
            
            save_checkpoint ({
                'i_iter': i_iter,
                'state_dict': model.state_dict (),
                'optimizer': optimizer.state_dict ()
                }, CHECKPOINT_SAVE_PATH + 'checkpoint_' + str (i_iter) + '.pth.tar')
        sys.stdout.flush()
        time.sleep (1.0)
        pbar.write (s ='ipoc ' +  str (i_ipoc) + ' iter ' + str (i_iter) + ' loss ' + str (ipoc_loss))


def pred_func (raw, factor):
    '''
        raw: full resolution ()
    '''
    origin_size = raw.shape ()[1::] #HxW
    raw = raw [::, ::factor, ::factor]
    raw = torch.tensor (np.expand_dims (raw, 0), device=device, dtype=torch.float32) / 255.0
    pred = refineNet (raw)
    pred = np.squeeze (pred.cpu ().numpy ()) # 2 dimensional pred
    pred = resize (pred, origin_size, order=0, mode='wrap', preserve_range=True) #Back to origin size
    pred = (pred * 255).astype (np.uint8)
    return pred



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument ('--gpu',    help='comma seperated list of GPU(s) to use.')
    parser.add_argument ('--size', help='size of input data', required=True)
    parser.add_argument ('--factor', help='downsample factor of data', required=True)
    parser.add_argument ('--load',    help='load model')
    parser.add_argument ('--batch_size', default='4')
    parser.add_argument ('--refine')
    
    

    args = parser.parse_args()
    checkpoint_path = None

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DOWNSAMPLE_FACTOR = int (args.factor)
    LOG_DIR = LOG_DIR + args.size + '_' + str (DOWNSAMPLE_FACTOR) + '/'
    CHECKPOINT_SAVE_PATH = CHECKPOINT_SAVE_PATH + args.size + '_' + str (DOWNSAMPLE_FACTOR) + '/'
    if ()
    print ('CHECKPOINT_SAVE_PATH:', CHECKPOINT_SAVE_PATH)
    print ('LOG_DIR path', LOG_DIR)


    create_dir (CHECKPOINT_SAVE_PATH)
    size = int (args.size)
    INPUT_SHAPE = (1, size, size)

    BATCH_SIZE = int (args.batch_size)

    # GPU setup
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.load:
        checkpoint_path = args.load

    print ('Using GPU', os.environ['CUDA_VISIBLE_DEVICES'])    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RefineNet setup
    if args.refine is not None:
        refineNet = FNC2D (IN_CH, OUT_CH).to (device)
        refineCheckpoint = torch.load (args.refine)
        refineNet.load_state_dict (refineCheckpoint ['state_dict'])

    # Setup dataflow
    X_train, y_train = get_data (DOWNSAMPLE_FACTOR)
    cremiData = CremiDataset ('train', X_train, y_train, sample_size=(size, size))
    train_data = DataLoader (cremiData, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Setup model
    model = FCN2D (IN_CH, OUT_CH).to (device)
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
