import os, time
from natsort import natsorted
import torch
import argparse
from models import FusionNet
from train import train

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]

parser = argparse.ArgumentParser(description='FusionNet joint train refine')

subparsers = parser.add_subparsers(help='sub-command type')

parser_spliter = subparsers.add_parser('spliter', help='spliter')
parser_spliter.add_argument ('--weight', type=float, default=None, nargs='+', metavar=('a', 'b'))
parser_spliter.add_argument ('--dilation', type=int, default=0)
parser_spliter.add_argument ('--erosion', type=int, default=2)
parser_spliter.add_argument ('--size', type=int, default=[128, 128], nargs='+')
parser_spliter.add_argument ('-t', default='spliter', dest="type")


parser_merger = subparsers.add_parser('merger', help='merger')
parser_merger.add_argument ('--weight', type=float, default=None, nargs='+', metavar=('a', 'b'))
parser_merger.add_argument ('--dilation', type=int, default=1)
parser_merger.add_argument ('--erosion', type=int, default=0)
parser_merger.add_argument ('--size', type=int, default=[128, 128], nargs='+')
parser_merger.add_argument ('-t', default='merger', dest="type")


parser.add_argument ('--gpu-id', type=int, help='choose GPU to use.', required=True)
parser.add_argument ('--batch_size', type=int, default=12)
parser.add_argument ('--load')
parser.add_argument ('--loss', choices=['WBCE', 'BCE'], default='BCE')

args = parser.parse_args()

if __name__ == '__main__':
    print ("using gpu-ids:", args.gpu_id)
    gpu_id = args.gpu_id

    with torch.cuda.device (gpu_id):
        model = FusionNet (IN_CH, FEATURES, OUT_CH)
        if args.load:
            checkpoint = torch.load  (args.load, map_location='cpu')
            model.load_state_dict (checkpoint['state_dict'])
        model = model.cuda ()
    
    print (args)
    name = args.type + '_' + args.loss + '_' + str (args.size [0]) + '_' + str (args.dilation) + str (args.erosion)
    if args.weight is not None:
        weight0_str = str (args.weight [0])
        if (len (weight0_str) > 5):
            weight0_str = weight0_str [0:5]
        name = name + '_' + weight0_str 

    print ('Training: ', args.type, '\tloss: ' + args.loss)
    if args.weight is not None:
        print ('weight: ', args.weight)
    print ('dilation: ', args.dilation, '\terosion: ', args.erosion)
    print ('output size: ', args.size)

    train (model, args, name)
