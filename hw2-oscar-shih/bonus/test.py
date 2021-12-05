import os
import argparse
import glob
from dataset import myDataset
import torch
from torch.backends import cudnn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from model import DSN
from PIL import Image
import random 
import numpy as np

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    model = DSN().cuda()
    same_seeds(1126)
    state = torch.load(config.ckp_path)
    model.load_state_dict(state['state_dict'])

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)

    out_filename = config.save_path
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_name,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn).convert('RGB')
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.cuda()
                output, _, _, _, _ = model(data, mode=config.mode)
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw2_data/digits/svhn/test')
    parser.add_argument('--save_path', type=str, default='ckpt/test')
    parser.add_argument('--ckp_path', default='ckpt/test/3750-dann.pth', type=str, help='Checkpoint path.')
    parser.add_argument('--mode', default='target', type=str, help='mode for model')
    

    config = parser.parse_args()
    print(config)
    main(config)