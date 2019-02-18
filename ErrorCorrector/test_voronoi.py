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
from utils import *


IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]
continuous = True

gpu_id = 0
def setup_env_conf ():
    spliter = spliter_thres_fn
    merger = merger_thres_fn

    env_conf = {
        "corrector_size": [96, 96], 
        "spliter": spliter,
        "merger": merger,
        "cell_thres": int (255 * 0.5),
        "T": 6,
        "agent_out_shape": [1, 4, 4],
        "observation_shape": [3, 256, 256],
        "env_gpu": 0,
        "reward_thres": 0.5,
        "num_segs": 40,
        "merge_err": True,
        "split_err": False,
        "continuous": continuous,
        "use_stop": False,
        "num_err": 6,
        "alpha": 5,
        "beta": 2,
    }
    if env_conf ["split_err"]:
        env_conf ["spliter"] = merger_thres_fn
        print ("use merger_thres")
    env_conf ["num_action"] = int (np.prod (env_conf ['agent_out_shape']))
    if env_conf ["continuous"]:
        env_conf ["num_action"] = 2
    env_conf ["num_feature"] = env_conf ['observation_shape'][0]
    return env_conf

env_config = setup_env_conf ()
env = Voronoi_env (env_config)

def obs2tensor (obs):
    ret = obs [None]
    if gpu_id >= 0:
        ret = torch.tensor (ret, dtype=torch.float32).cuda ()
    return ret

def setup_rl_model (env, env_conf):
    if env_conf ["continuous"]:
        model = A3Clstm_continuous (env_conf ["observation_shape"], 
                        env_conf["num_action"], 512)
    else:
        model = A3Clstm (env_conf ["observation_shape"], 
                            env_conf["num_action"], 512)
    if gpu_id >= 0:
        model = model.cuda ()
    return model


if gpu_id >= 0:
    cx = Variable(torch.zeros(1, 512).cuda())
    hx = Variable(torch.zeros(1, 512).cuda())
else:
    cx = Variable(torch.zeros(1, 512))
    hx = Variable(torch.zeros(1, 512))

done = False
cnt = 0

if not continuous:
    for i in range (env.agent_out_shape [1]):
        for j in range (env.agent_out_shape [2]):
            print (str (cnt), end='\t', flush=True)
            cnt += 1
        print ()

done = False
obs = env.reset ()
print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()

model = setup_rl_model (env, env_config)
sum_score = 0

print (obs.shape)

while not done:
    
    obs_t = obs2tensor (obs)
    if not continuous:
        value, logit, (hx, cx) = model ((Variable (obs_t), (hx, cx)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        action = prob.max (1)[1].data.cpu ().numpy () [0]
        action = int (input ('action = '))
        prob_np = prob.data.cpu ().numpy ()
        print ("______________________________")
        print ("Prob: ")
        print (prob_np)
        action_index = env.int2index (action, env.agent_out_shape)
        error_index = env.index2validrange (action_index [1:], env.agent_out_shape [1:])
        print ("action: ", action)
        print ("action index: ", action_index)
        print ("error index: ", error_index)
    else:
        value, mu, sigma, (hx, cx) = model (
            (Variable(obs_t), (hx, cx)))
        mu = torch.clamp(mu.data, -1.0, 1.0)

        sigma = sigma + 1e-3
        eps = torch.randn (mu.size()).cuda ()
        eps = Variable (eps)
        action = (mu + sigma.sqrt () * eps).data
        action = torch.clamp (action, -1.0, 1.0)
        

        act = Variable (action)
        prob = normal (act, mu, sigma, gpu_id, gpu=gpu_id >= 0)
        action = action.cpu().numpy()[0]

        # y_apx = float (input ("y_apx = "))
        # x_apx = float (input ("x_apx = "))
        # action = [y_apx, x_apx]

        y_apx, x_apx = action [0], action [1]
        error_index = env.approx2index (y_apx, x_apx, env.raw.shape)
        print ("mu: ", mu, "sigma", sigma)
        print ("action_apx", y_apx, x_apx)
        print ("Action: ", error_index)
        print ("Prob: ", prob)



    cnt = 0
    
    obs, reward, done, info = env.step (action)
    tmp = []
    for c in range (len (obs)):
        tmp += [obs [c]]
    tmp = np.concatenate (tmp, 1)
    print ("reward:", reward)
    print ("old_score:", env.old_score)
    print ("done: ", done)
    plt.imshow (tmp, cmap='gray')
    plt.show ()
    sum_score += reward
    # plt.imshow (env.render ())
    # plt.show ()
    # print ('action: ', action) 
    # print ('reward: ', reward)

print ("old_score", env.old_score)
plt.imshow (env.render ())
plt.show ()
