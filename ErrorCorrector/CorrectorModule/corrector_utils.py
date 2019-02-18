import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from CorrectorModule.models import FusionNet, UNet
import sys
sys.path.append('../')

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

# spliter_net = None
# merger_net = None

def init_model (model, gpu_id, checkpoint_path):
    checkpoint = torch.load  (checkpoint_path, map_location='cpu')
    model.load_state_dict (checkpoint ['state_dict'])
    # model.share_memory()
    model.eval ()
    with torch.cuda.device (gpu_id):
        model = model.cuda ()
    return model

def spliter_FusionNet_fn (gpu_id): 
    spliter_checkpoint = 'CorrectorModule/checkpoints/spliter_WBCE_96_02/checkpoint_816000.pth.tar'
    spliter_net = FusionNet (IN_CH, FEATURES, OUT_CH)
    spliter_net = init_model (spliter_net, gpu_id, spliter_checkpoint)
    
    def spliter_FusionNet (raw, prob, gpu_id, spliter_net=spliter_net):
        with torch.no_grad ():
            with torch.cuda.device (gpu_id):
                mask = np.zeros_like (raw)
                x = np.concatenate ([raw[None][None], mask[None][None]], 1)
                x_t = torch.tensor (x, dtype=torch.float32).cuda ()
                x_t = x_t / 255.0 
                ret_t = spliter_net (x_t)
                ret = ret_t.cpu ().numpy ()[0][0]
        return ret * 255.0
    return spliter_FusionNet
    
def merger_FusionNet_fn (gpu_id):
    # global merger_net
    # if merger_net is None:
    merger_checkponit = 'CorrectorModule/checkpoints/UNet_normal_WBCE_96_00/checkpoint_501000.pth.tar'
    merger_net = FusionNet (IN_CH, FEATURES, OUT_CH)
    merger_net = init_model (merger_net, gpu_id, merger_checkponit)

    def merger_FusionNet (raw, prob, gpu_id, merger_net=merger_net):
        with torch.no_grad ():
            with torch.cuda.device (gpu_id):
                mask = np.zeros_like (raw)
                x = np.concatenate ([raw[None][None], mask[None][None]], 1)
                x_t = torch.tensor (x, dtype=torch.float32).cuda ()
                x_t = x_t / 255.0 
                ret_t = merger_net (x_t)
                ret = ret_t.cpu ().numpy ()[0][0]
        return ret * 255.0
    return merger_FusionNet

def spliter_thres_fn (gpu_id=None):
    def spliter_thres (raw, prob, gpu_id):
        ret = (prob > (255 * 0.7)).astype (np.float32) * 255.0
        return ret
    return spliter_thres

def merger_thres_fn (gpu_id=None):
    def merger_thres (raw, prob, gpu_id):
        ret = (prob > (255 * 0.3)).astype (np.float32) * 255.0
        return ret
    return merger_thres

def unet_96_fn (gpu_id):
    checkponit = 'CorrectorModule/checkpoints/UNet_normal_WBCE_96_02/checkpoint_321000.pth.tar'
    net = UNet (IN_CH, FEATURES, OUT_CH)
    net = init_model (net, gpu_id, checkponit)

    def unet_96 (raw, prob, gpu_id, net=net):
        with torch.no_grad ():
            with torch.cuda.device (gpu_id):
                mask = np.zeros_like (raw)
                x = np.concatenate ([raw[None][None], mask[None][None]], 1)
                x_t = torch.tensor (x, dtype=torch.float32).cuda ()
                x_t = x_t / 255.0 
                ret_t = net (x_t)
                ret = ret_t.cpu ().numpy ()[0][0]
        return ret * 255.0
    return unet_96

def unet_160_fn (gpu_id):
    checkponit = 'CorrectorModule/checkpoints/UNet_normal_WBCE_96_02/checkpoint_321000.pth.tar'
    net = UNet (IN_CH, FEATURES, OUT_CH)
    net = init_model (net, gpu_id, checkponit)

    def unet_160 (raw, prob, gpu_id, net=net):
        with torch.no_grad ():
            with torch.cuda.device (gpu_id):
                mask = np.zeros_like (raw)
                x = np.concatenate ([raw[None][None], mask[None][None]], 1)
                x_t = torch.tensor (x, dtype=torch.float32).cuda ()
                x_t = x_t / 255.0 
                ret_t = net (x_t)
                ret = ret_t.cpu ().numpy ()[0][0]
        return ret * 255.0
    return unet_160