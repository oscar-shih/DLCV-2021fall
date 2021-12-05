import os
import argparse
from solver import Solver
from dataset import myDataset, myDataset_2
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


def main(config):
    # For fast training.
    cudnn.benchmark = True

    if config.use_wandb:
        import wandb
        wandb.init(project="dlcv-hw3-4", config=config)
        config = wandb.config
        print(config)

    label_path = os.path.join(config.data_path, config.src_domain, 'train.csv')
    src_root = os.path.join(config.data_path, config.src_domain, 'train')

    src_label_data = []
    with open(label_path) as f:
        src_label_data += f.readlines()[1:]

    src_label_train, src_label_val = train_test_split(src_label_data, test_size=0.25, shuffle=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    transform_aug = transforms.Compose([
        transforms.ColorJitter(
            brightness=63.0 / 255.0, saturation=[0.5, 1.5], contrast=[0.2, 1.8]
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    if config.data_aug:
        src_trainset = myDataset(root=src_root, label_data=src_label_train, transform=transform_aug)

    else:
        src_trainset = myDataset_2(root=src_root, label_data=src_label_train, transform=transform)

    src_trainset_loader = DataLoader(src_trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    src_valset = myDataset_2(root=src_root, label_data=src_label_val, transform=transform)
    src_valset_loader = DataLoader(src_valset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    if config.src_only:
        solver = Solver(src_trainset_loader, src_valset_loader, None, config)

    else:
        tgt_root = os.path.join(config.data_path, config.tgt_domain, 'train')
        tgt_trainset = myDataset_2(root=tgt_root, transform=transform)

        tgt_trainset_loader = DataLoader(tgt_trainset, batch_size=config.batch_size, shuffle=True, num_workers=4)

        solver = Solver(src_trainset_loader, src_valset_loader, tgt_trainset_loader, config)


    solver.train()
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # path.
    parser.add_argument('--data_path', type=str, default='../hw2_data/digits', help="the path of the dataset to train.")
    parser.add_argument('--ckp_dir', default='ckpt/', type=str, help='Checkpoint path', required=False)
    parser.add_argument('--name', default='', type=str, help='Name for saving model')
    parser.add_argument('--src_domain', default='mnistm', type=str, choices=['mnistm', 'svhn', 'usps'], help='Name for source domain')
    parser.add_argument('--tgt_domain', default='svhn', type=str, choices=['mnistm', 'svhn', 'usps'], help='Name for target domain')

    # training configuration.
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=30000, help='number of total iterations')
    parser.add_argument('--num_iters_decay', type=int, default=20000, help='number of iterations for decaying lr')
    parser.add_argument('--step_decay_weight', type=float, default=0.9, help='multiplicative factor of learning rate decay')
    parser.add_argument('--active_domain_loss_step', type=int, default=10000, help='number of iterations for activating any additional domain adaptation loss')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument("--lr", help="the learning rate", default=1e-3, type=float)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay for Adam optimizer')
    parser.add_argument('--alpha_weight', type=float, default=0.015, help='weight for recon loss')
    parser.add_argument('--beta_weight', type=float, default=0.075, help='weight for diff loss')
    parser.add_argument('--gamma_weight', type=float, default=0.25, help='weight for sim loss')
    parser.add_argument("--src_only", help="only train on src data", action="store_true")
    parser.add_argument("--data_aug", help="data augmentation", action="store_true")

    # Step size.
    parser.add_argument('--log_interval', type=int, default=250)
    parser.add_argument('--save_interval', type=int, default=1000)

    # Others
    parser.add_argument("--use_wandb", help="log training with wandb, "
        "requires wandb, install with \"pip install wandb\"", action="store_true")
    

    config = parser.parse_args()
    print(config)
    main(config)
