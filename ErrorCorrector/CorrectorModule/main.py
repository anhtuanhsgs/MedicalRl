import os, time
from natsort import natsorted
import torch
import argparse
from models import FusionNet, UNet
from train import train
from deploy import deploy

IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128, 256]

parser = argparse.ArgumentParser(description='FusionNet joint train refine')

subparsers = parser.add_subparsers(help='sub-command type')

parser_normal = subparsers.add_parser('normal', help='normal')
parser_normal.add_argument ('--weight', type=float, default=None, nargs='+', metavar=('a', 'b'))
parser_normal.add_argument ('--dilation', type=int, default=1)
parser_normal.add_argument ('--erosion', type=int, default=0)
parser_normal.add_argument ('--size', type=int, default=[128, 128], nargs='+')
parser_normal.add_argument ('-t', default='normal', dest="type")

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

parser_deploy = subparsers.add_parser ("deploy", help="Deploy")
parser_deploy.add_argument ("--data_path", type=str, default="../Data/train/A/train-input.tif")
parser_deploy.add_argument ("--save_path", type=str, default=None)
parser_deploy.add_argument ('--size', type=int, default=[96, 96], nargs='+')
parser_deploy.add_argument ("--data-type", type=str, default="train")
parser_deploy.add_argument ("--name", type=str, default="deploy")
parser_deploy.add_argument ('-t', default='deploy', dest="type")
    
parser.add_argument ('--gpu-id', type=int, help='choose GPU to use.', required=True)
parser.add_argument ('--batch_size', type=int, default=12)
# parser.add_argument ('--load', default='checkpoints/spliter_WBCE_96_00/checkpoint_806000.pth.tar')
parser.add_argument ('--load', default=None)
parser.add_argument ('--loss', choices=['WBCE', 'BCE'], default='BCE')
parser.add_argument ('--model', choices=['UNet', "Fusion"], default='UNet')

args = parser.parse_args()

if __name__ == '__main__':
    print ("using gpu-ids:", args.gpu_id)
    gpu_id = args.gpu_id

    with torch.cuda.device (gpu_id):
        if args.model == "Fusion":
            model = FusionNet (IN_CH, FEATURES, OUT_CH)
        else:
            model = UNet (IN_CH, FEATURES, OUT_CH)

        if args.load:
            checkpoint = torch.load  (args.load, map_location='cpu')
            model.load_state_dict (checkpoint['state_dict'])

        model = model.cuda ()
    
    if args.type != "deploy":
        print (args)
        name = args.model + "_" + args.type + '_' + args.loss + '_' + \
                    str (args.size [0]) + '_' + str (args.dilation) + str (args.erosion)
        # weight0_str = str (args.weight [0])
        # if (len (weight0_str) > 5):
        #     weight0_str = weight0_str [0:5]
        # name = name + '_' + weight0_str 

        print ('Training: ', args.type, '\tloss: ' + args.loss)
        if args.weight is not None:
            print ('weight: ', args.weight)
        print ('dilation: ', args.dilation, '\terosion: ', args.erosion)
        print ('output size: ', args.size)

        train (model, args, name)
    
    else:
        print ("Deploy!\n")
        assert (args.load != None)
        if args.save_path is None:
            if args.data_type != "train":
                args.data_path = "../Data/test/A/test-input.tif"
                args.save_path = "../Data/test/deploy/" + args.name + ".tif"
            else:
                args.data_path = "../Data/train/A/train-input.tif"
                args.save_path = "../Data/train/deploy/" + args.name + ".tif"
        deploy (model, args)

