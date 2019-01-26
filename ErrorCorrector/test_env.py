from environment import *
from Utils.img_aug_func import *
from natsort import natsorted
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from CorrectorModule.models import FusionNet
from CorrectorModule.corrector_utils import *

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

env_config = {
    "corrector_size": [96, 96], 
    "spliter": spliter_thres,
    "merger": merger_thres,
    "cell_thres": int (255 * 0.5),
    "T": 16,
    "agent_out_shape": [2, 8, 8],
    "num_feature": 6,
    "num_action": 2 * 8 * 8,
    "observation_shape": [6, 256, 256],
    "env_gpu": 3
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

# print ('test rand_score:', rand_score (gt_lbl [0], gt_lbl [0]))

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
env = EM_env (raw, lbl, prob, env_config, 'train', gt_lbl)

done = False
obs = env.reset ()

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()
cnt = 0
for i in range (env.agent_out_shape [1]):
    for j in range (env.agent_out_shape [2]):
        print (str (cnt), end='\t', flush=True)
        cnt += 1
    print ()

sum_score = 0
while not done:
    action = int (input ('a = '))

    action_index = env.int2index (action, env.agent_out_shape)
    error_index = env.index2validrange (action_index [1:], env.agent_out_shape [1:])
    print ("action index", action_index)
    print ("error index", error_index)

    cnt = 0
    
    
    obs, reward, done, info = env.step (action)
    tmp = []
    for c in range (len (obs)):
        tmp += [obs [c]]
    tmp = np.concatenate (tmp, 1)
    print ("reward:", reward)
    print ("old_score:", env.old_score)
    plt.imshow (env.render ())
    plt.show ()
    sum_score += reward
    # plt.imshow (env.render ())
    # plt.show ()
    # print ('action: ', action) 
    # print ('reward: ', reward)

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

