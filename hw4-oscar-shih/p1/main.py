import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

from dataset import MiniDataset
from sampler import GeneratorSampler, NShotTaskSampler
from solver import Solver

def randomseed(seed):
    random.seed(seed)
        # Numpy
    np.random.seed(seed)
        # Torch
    torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")

    # training configuration.
    parser.add_argument('--episodes_per_epoch', default=600, type=int, help='episodes per epoch')
    parser.add_argument('--N_way_train', default=5, type=int, help='N_way (default: 5) for training')
    parser.add_argument('--N_shot_train', default=1, type=int, help='N_shot (default: 1) for training')
    parser.add_argument('--N_query_train', default=15, type=int, help='N_query (default: 15) for training')
    parser.add_argument('--N_way_val', default=5, type=int, help='N_way (default: 5) for val')
    parser.add_argument('--N_shot_val', default=1, type=int, help='N_shot (default: 1) for val')
    parser.add_argument('--N_query_val', default=15, type=int, help='N_query (default: 15) for val')
    parser.add_argument('--matching_fn', default='l2', type=str, choices=['l2', 'cosine', 'parametric'], help='distance matching function')

    # optimizer configuration
    parser.add_argument("--lr", help="the learning rate", default=1e-4, type=float)
    parser.add_argument('--num_steps_decay', type=int, default=40, help='number of steps for decaying lr')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight_decay for Adam optimizer')

    # path.
    parser.add_argument('--train_csv', type=str, default='../hw4_data/mini/train.csv', help="Training images csv file")
    parser.add_argument('--train_data_dir', type=str, default='../hw4_data/mini/train/', help="Training images directory")
    parser.add_argument('--val_csv', type=str, default='../hw4_data/mini/val.csv', help="val images csv file")
    parser.add_argument('--val_data_dir', type=str, default='../hw4_data/mini/val/', help="val images directory")
    parser.add_argument('--val_testcase_csv', type=str, default='../hw4_data/mini/val_testcase.csv', help="val test case csv")
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')

    # Step size.
    parser.add_argument('--num_epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this epoch')
    parser.add_argument('--log_interval', type=int, default=300)
    parser.add_argument('--ckp_interval', type=int, default=600)

    return parser.parse_args()

if __name__=='__main__':

    randomseed(1126)
    args = parse_args()

    train_dataset = MiniDataset(args.train_csv, args.train_data_dir)
    val_dataset = MiniDataset(args.val_csv, args.val_data_dir)

    train_loader = DataLoader(
        train_dataset,
        num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
        batch_sampler=NShotTaskSampler(args.train_csv, args.episodes_per_epoch, args.N_way_train, args.N_shot_train, args.N_query_train)
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.N_way_val * (args.N_query_val + args.N_shot_val),
        num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.val_testcase_csv)
    )

    solver = Solver(args, train_loader, val_loader)

    solver.train()