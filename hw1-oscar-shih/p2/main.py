import argparse
from solver import Solver
from dataset import TrainDataset
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import random
import numpy as np
import torch
from torch.backends import cudnn

# def same_seeds(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     cudnn.benchmark = True
#     cudnn.deterministic = True

# same_seeds(8763)

def main(config):
    # For fast training.
    cudnn.deterministic = True
    cudnn.benchmark = True

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_set = TrainDataset(root=config.trainset_dir, transform=tfm)
    valid_set = TrainDataset(root=config.valset_dir, transform=tfm)

    # Data loader.
    train_set_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    solver = Solver(train_set_loader, valid_set_loader, config)
    solver.train()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model = DeepLabv3_ResNet50()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True, min_lr=1e-5)

    # Training configuration.
    parser.add_argument('--trainset_dir', type=str, default='../hw1_data/p2_data/train')
    parser.add_argument('--valset_dir', type=str, default='../hw1_data/p2_data/validation')
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=30, help='number of total epoch')
    parser.add_argument('--resume_iter', type=int, default=0, help='resume training from this iteration')
    parser.add_argument('--name', default='', type=str, help='Name for saving model')
    parser.add_argument('--model_type', default='DeepLabv3_Resnet50', choices=['FCN32s', 'FSN8s', 'DeepLabv3_Resnet50'], type=str, help='Model type')
    parser.add_argument('--ckp_dir', default='../ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=500)
    

    config = parser.parse_args()
    print(config)
    main(config)