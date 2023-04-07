import os
import time
import logging
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torch.multiprocessing as mp
from utils.util import setup_logger, print_args
from models import Trainer
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description='referenceSR Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--phase', default='test', type=str)
    parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
    parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
    parser.add_argument('--n_resblocks', type=int, default=20,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=128,
                        help='number of feature maps')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')

    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')

    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='5', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--low_rank_tensor', default=4, type=int)
    ## network setting
    parser.add_argument('--net_name', default='CADN', type=str, help='')
    parser.add_argument('--sr_scale', default=4, type=int)
    parser.add_argument('--input_nc', default=3, type=int)
    parser.add_argument('--output_nc', default=3, type=int)
    parser.add_argument('--nf', default=64, type=int)
    parser.add_argument('--n_blks', default='4, 4, 4', type=str)
    parser.add_argument('--nf_ctt', default=32, type=int)
    parser.add_argument('--n_blks_ctt', default='2, 2, 2', type=str)
    parser.add_argument('--num_nbr', default=1, type=int)
    parser.add_argument('--n_blks_dec', default=10, type=int)
    parser.add_argument('--ref_level', default=1, type=int)

    ## dataloader setting
    parser.add_argument('--data_root', default='/data/liuyu/',type=str)
    parser.add_argument('--dataset', default='CUFED', type=str, help='CUFED')
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_augmentation', default=False, type=bool)
    
    parser.add_argument('--resume', default='./pretrained_weights/final_rec.pth', type=str)
    parser.add_argument('--testset', default='TestSet_multi', type=str, help='Sun80 | Urban100 | TestSet_multi')
    parser.add_argument('--save_folder', default='./test_results/', type=str)


    ## setup training environment
    args = parser.parse_args()
    def init_dist(backend='nccl', **kwargs):
        """initialization for distributed training"""
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn')
        rank = int(os.environ["RANK"])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        dist.init_process_group(backend=backend, **kwargs)
    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    args.save_folder = os.path.join(args.save_folder, args.testset, args.name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    log_file_path = args.save_folder + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    setup_logger(log_file_path)

    print_args(args)
    cudnn.benchmark = True

    ## test model
    trainer = Trainer(args)
    trainer.test()


if __name__ == '__main__':
    main()
