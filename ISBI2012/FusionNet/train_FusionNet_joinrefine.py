import os, time
os.environ["OMP_NUM_THREADS"] = "1"
from natsort import natsorted
import torch
import torch.multiprocessing as mp
import argparse
from FusionNet import FusionNet
from train_joinrefine import train

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

if __name__ == '__main__':
    mp.set_start_method('spawn')
    models_paths = [
        'checkpoints/0_1.0/checkpoint_90375.pth.tar',
        'checkpoints/2_1.0/checkpoint_154254.pth.tar',
        'checkpoints/1_1.0/checkpoint_112048.pth.tar',
        'checkpoints/3_1.0/checkpoint_163372.pth.tar'
    ]

    parser = argparse.ArgumentParser(description='FusionNet joint train refine')
    parser.add_argument ('--gpu-id', type=int, help='comma seperated list of GPU(s) to use.', nargs='+')
    parser.add_argument ('--zoom_factor', type=float, help='zoom-in factor across level', default=0.6, metavar='G')
    parser.add_argument ('--batch_size', type=int, nargs='+')
    
    args = parser.parse_args()

    print ("using gpu-ids:", args.gpu_id)

    models = []
    for level in range (4):
        gpu_id = args.gpu_id [level % len (args.gpu_id)]
        with torch.cuda.device (gpu_id):
            model = FusionNet (IN_CH, FEATURES, OUT_CH)
            checkpoint = torch.load  (models_paths [level], map_location='cpu')
            model.load_state_dict (checkpoint['state_dict'])
            model = model.cuda ()
            model.share_memory ()
            models.append (model)

    processes = []
    for level in range (4):
        p = mp.Process (
            target=train, args=(models, args, level,))
        p.start ()
        processes.append (p)
        time.sleep (0.1)

    for p in processes:
        time.sleep (0.1)
        p.join ()