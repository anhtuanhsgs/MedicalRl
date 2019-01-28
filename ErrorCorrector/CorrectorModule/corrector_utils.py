import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from CorrectorModule.models import FusionNet
import sys
sys.path.append('../')

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

spliter_net = None
merger_net = None

def init_fusion (model, gpu_id, checkpoint_path):
    with torch.cuda.device (gpu_id):
        model = model.cuda ()
        checkpoint = torch.load  (checkpoint_path)
        model.load_state_dict (checkpoint ['state_dict'])
        model.share_memory()
        model.eval ()

def spliter_FusionNet (raw, prob, gpu_id):
    global spliter_net
    if spliter_net is None:
        spliter_checkpoint = '../CorrectorModule/checkpoints/spliter_WBCE_128_02_0.95/checkpoint_602000.pth.tar'
        spliter_net = FusionNet (IN_CH, FEATURES, OUT_CH)
        init_fusion (spliter_net, gpu_id, spliter_checkpoint)

    with torch.no_grad ():
        with torch.cuda.device (gpu_id):
            mask = np.zeros_like (raw)
            x = np.concatenate ([raw[None][None], mask[None][None]], 1)
            x_t = torch.tensor (x, dtype=torch.float32).cuda ()
            x_t = x_t / 255.0 
            ret_t = spliter_net (x_t)
            ret = ret_t.cpu ().numpy ()[0][0]
    return ret * 255.0
    
def merger_FusionNet (raw, prob, gpu_id):
    global merger_net
    if merger_net is None:
        merger_checkponit = '../CorrectorModule/checkpoints/merger_WBCE_128_00_0.2/checkpoint_592000.pth.tar'
        merger_net = FusionNet (IN_CH, FEATURES, OUT_CH)
        init_fusion (merger_net, gpu_id, merger_checkponit)

    with torch.no_grad ():
        with torch.cuda.device (gpu_id):
            mask = np.zeros_like (raw)
            x = np.concatenate ([raw[None][None], mask[None][None]], 1)
            x_t = torch.tensor (x, dtype=torch.float32).cuda ()
            x_t = x_t / 255.0 
            ret_t = merger_net (x_t)
            ret = ret_t.cpu ().numpy ()[0][0]
    return ret * 255.0

def spliter_thres (raw, prob, gpu_id):
    ret = (prob > (255 * 0.85)).astype (np.float32) * 255.0
    return ret

def merger_thres (raw, prob, gpu_id):
    ret = (prob > (255 * 0.55)).astype (np.float32) * 255.0
    return ret