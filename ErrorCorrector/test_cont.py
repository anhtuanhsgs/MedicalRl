from environment_continuous import *
from Utils.img_aug_func import *
from natsort import natsorted
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from CorrectorModule.models import FusionNet

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

spliter_net = FusionNet (IN_CH, FEATURES, OUT_CH)
merger_net = FusionNet (IN_CH, FEATURES, OUT_CH)

with torch.cuda.device (0):
    spliter_net = spliter_net.cuda ()
    merger_net = merger_net.cuda ()
    spliter_checkpoint = 'CorrectorModule/checkpoints/spliter_WBCE_128_02_0.95/checkpoint_602000.pth.tar'
    merger_checkponit = 'CorrectorModule/checkpoints/merger_WBCE_128_00_0.2/checkpoint_592000.pth.tar'
    spl_checkpoint = torch.load  (spliter_checkpoint)
    mer_checkpoint = torch.load (merger_checkponit)
    spliter_net.load_state_dict (spl_checkpoint ['state_dict'])
    merger_net.load_state_dict (mer_checkpoint ['state_dict'])


def spliter2 (raw, prob):
    with torch.no_grad ():
        with torch.cuda.device (0):
            mask = np.zeros_like (raw)
            x = np.concatenate ([raw[None][None], mask[None][None]], 1)
            x_t = torch.tensor (x, dtype=torch.float32).cuda ()
            x_t = x_t / 255.0 
            ret_t = spliter_net (x_t)
            ret = ret_t.cpu ().numpy ()[0][0]
    return ret * 255.0

def merger2 (raw, prob):
    with torch.no_grad ():
        with torch.cuda.device (0):
            mask = np.zeros_like (raw)
            x = np.concatenate ([raw[None][None], mask[None][None]], 1)
            x_t = torch.tensor (x, dtype=torch.float32).cuda ()
            x_t = x_t / 255.0 
            ret_t = merger_net (x_t)
            ret = ret_t.cpu ().numpy ()[0][0]
    return ret * 255.0

def spliter (raw, prob):
    ret = (prob > (255 * 0.85)).astype (np.float32) * 255.0
    return ret

def merger (raw, prob):
    ret = (prob > (255 * 0.55)).astype (np.float32) * 255.0
    return ret

env_config = {
    'corrector_size': [256, 256], 
    'spliter': spliter2,
    'merger': merger2,
    'cell_thres': int (255 * 0.5),
    'T': 100,
    'agent_out_shape': [2, 16, 16],
    'num_feature': 3,
    'continuous': True
}


def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train


raw , gt_lbl = get_data (path='Data/train/', args=None)
prob = io.imread ('Data/train-membranes-idsia.tif')

raw = raw [:10]
gt_lbl = gt_lbl [:10]
prob = prob [:10]

print ('test rand_score:', rand_score (gt_lbl [0], gt_lbl [0]))

# prob_r = []
# for img in raw:
#     prob_r += [spliter2 (img, None)]
# prob_r = np.array (prob_r)
# prob = prob_r
# tmp = spliter2 (raw[1], None)
# plt.imshow (tmp)
# plt.show ()

lbl = []
for img in prob:
    lbl += [label (img > env_config ['cell_thres'])]
lbl = np.array (lbl)
print (lbl.shape)
env = EM_env (raw, lbl, prob, env_config, 'train', gt_lbl)

done = False
obs = env.reset ()

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

sum_score = 0

while not done:
    action = input ('(a, y, x) = ')
    action = action.split ()
    action = (int (action [0]), float (action[1]), float (action [2]))
    
    shape = env_config ['agent_out_shape']
    
    obs, reward, done, info = env.step (action)
    tmp = []
    for c in range (len (obs)):
        tmp += [obs [c]]
    tmp = np.concatenate (tmp, 1)
    plt.imshow (tmp)
    plt.show ()
    sum_score += reward
        # plt.imshow (env.render ())
        # plt.show ()
        # print ('action: ', action) 
        # print ('reward: ', reward)

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

