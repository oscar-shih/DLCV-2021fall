import glob
import os
import argparse
import random
import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import Model

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((196, 196)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    state = torch.load(config.ckpt_path)
    model = Model().to(device)
    model.load_state_dict(state['state_dict'])
    

    filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    filenames = sorted(filenames)
    out_filename = os.path.join(config.save_dir, 'test_pred.csv')
    os.makedirs(config.save_dir, exist_ok=True)
    
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn)
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='../hw3_data/p1_data/val')
    parser.add_argument('--save_dir', type=str, default='ckpt')
    parser.add_argument('--ckpt_path', default='ckpt/model3.ckpt', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    main(config)
 
