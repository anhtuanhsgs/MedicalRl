from environment import *
from Utils.img_aug_func import *
from natsort import natsorted
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import *

import torch
import torch.nn as nn
import torch.optim as optim

from CorrectorModule.models import FusionNet
from CorrectorModule.corrector_utils import *

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

gpu_id = 0
env_config = {
    "corrector_size": [160, 160], 
    "spliter": spliter_FusionNet,
    "merger": merger_FusionNet,
    "cell_thres": int (255 * 0.5),
    "T": 4,
    "agent_out_shape": [1, 2, 2],
    "num_feature": 2,
    "num_action": 1 * 2 * 2,
    "observation_shape": [2, 256, 256],
    "env_gpu": gpu_id
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

def obs2tensor (obs):
    ret = obs [None]
    if gpu_id >= 0:
        ret = torch.tensor (ret, dtype=torch.float32).cuda ()
    return ret

def setup_rl_model (env, env_conf):
    model = A3Clstm (env.observation_space.shape, env_conf["num_action"], 512)
    if gpu_id >= 0:
        model = model.cuda ()
    return model
#---------------------DATA-----------------------------
raw , gt_lbl = get_data (path='Data/train/', args=None)
prob = io.imread ('Data/train-membranes-idsia.tif')
raw = raw [:1]
gt_lbl = gt_lbl [:1]
prob = prob [:1]
prob = np.zeros_like (prob)
lbl = []
for img in prob:
    lbl += [label (img > env_config ['cell_thres'])]
lbl = np.array (lbl)
#---------------------DATA-----------------------------


env = EM_env (raw, lbl, prob, env_config, 'train', gt_lbl)


if gpu_id >= 0:
    cx = Variable(torch.zeros(1, 512).cuda())
    hx = Variable(torch.zeros(1, 512).cuda())
else:
    cx = Variable(torch.zeros(1, 512))
    hx = Variable(torch.zeros(1, 512))

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

model = setup_rl_model (env, env_config)

sum_score = 0
while not done:
    
    obs_t = obs2tensor (obs)
    value, logit, (hx, cx) = model ((Variable (obs_t), (hx, cx)))
    prob = F.softmax(logit, dim=1)
    log_prob = F.log_softmax(logit, dim=1)
    action = prob.max (1)[1].data.cpu ().numpy () [0]

    prob_np = prob.data.cpu ().numpy ()
    print ("______________________________")
    print ("Prob: ")
    print (prob_np)
    # for i in range (env_config ['agent_out_shape'][1]):
    #     for j in range (env_config ['agent_out_shape'][2]):
    #         print ("{:.3f}".format (prob_np [i,j]), end='\t')
    #     print ()


    action_index = env.int2index (action, env.agent_out_shape)
    error_index = env.index2validrange (action_index [1:], env.agent_out_shape [1:])
    print ("action: ", action)
    print ("action index: ", action_index)
    print ("error index: ", error_index)

    cnt = 0
    
    obs, reward, done, info = env.step (action)
    tmp = []
    for c in range (len (obs)):
        tmp += [obs [c]]
    tmp = np.concatenate (tmp, 1)
    print ("reward:", reward)
    print ("old_score:", env.old_score)
    print ("done: ", done)
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

