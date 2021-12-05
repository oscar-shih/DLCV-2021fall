import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader
import os
import argparse
import random
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default = "./generate_result", type = str)
parser.add_argument("--load_from", default = -1, type = int)
args = parser.parse_args()

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
images = []
randomseed(1126)

loader = ModelLoader(
    base_dir = '.',   
    name = 'default',
    load_from = args.load_from
)

os.makedirs(args.save_dir, exist_ok = True)

for i in range(1000):
    noise   = torch.randn(1, 512).cuda() 
    styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  
    images  = loader.styles_to_images(styles) 

    save_image(images, os.path.join(args.save_dir, f'{i}.jpg')) 
