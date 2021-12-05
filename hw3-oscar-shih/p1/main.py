from __future__ import print_function
import glob
import os
import random
from dataset import myDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from pytorch_pretrained_vit import ViT
import argparse
from dataset import myDataset
from solver import Solver
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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

randomseed(1126)

def main(config):
    train_tfm = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    ])

    valid_tfm = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    ])
    # load the dataset
    trainset = myDataset(root=config.trainset_dir, transform=train_tfm)
    valset = myDataset(root=config.valset_dir, transform=valid_tfm)
    a, b = valset[68]
    print(a.shape)
    print(b)
    print(len(trainset), len(valset))
    # Data loader.
    trainset_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    valset_loader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print(len(trainset_loader), len(valset_loader))

    solver = Solver(trainset_loader, valset_loader, config)
    solver.training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--trainset_dir', type=str, default='../hw3_data/p1_data/train')
    parser.add_argument('--valset_dir', type=str, default='../hw3_data/p1_data/val')
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of total epoch')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this iteration')
    parser.add_argument('--name', default='', type=str, help='Name for saving model')
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=920)
    
    config = parser.parse_args()
    print(config)
    main(config)
