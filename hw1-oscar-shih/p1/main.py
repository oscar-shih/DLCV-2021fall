import argparse
from solver import Solver
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
from dataset import image

import torch
import random
import numpy as np 


'''Random seed'''
def seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

seeds(7414)

'''Transforms'''
import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugment, AutoAugmentPolicy

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 224)
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(p=0.3),
    transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    transforms.ColorJitter(brightness=0.4),
    transforms.RandomRotation((-15,15)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

valid_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 224)
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
])

def main(config):
    # load the dataset
    train_set = image(root = 'hw1_data/p1_data/train_50', transform = train_tfm)
    valid_set = image(root = 'hw1_data/p1_data/val_50', transform = valid_tfm)

    # Data loader.
    trainset_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valset_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    solver = Solver(trainset_loader, valset_loader, config)
    solver.train()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=100, help='mini-batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of total epoch')
    parser.add_argument('--ckp_dir', default='ckpt', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this iteration') 
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=900)
    parser.add_argument("--data_aug", help="data augmentation", action="store_true")
    parser.add_argument('--name', default='', type=str, help='Name for saving model')
    config = parser.parse_args()
    print(config)
    main(config)
