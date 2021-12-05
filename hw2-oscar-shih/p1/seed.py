import torch
from torchvision.utils import save_image, make_grid
from stylegan2_pytorch import ModelLoader
import os
import argparse
import matplotlib.pyplot as plt
import random
import numpy as np
loader = ModelLoader(
    base_dir = '.',   
    name = 'default',
    load_from = 65
)
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

for i in range(32):
    noise   = torch.randn(1, 512).cuda() 
    styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  
    image  = loader.styles_to_images(styles) 
    images.append(image.squeeze(dim = 0).cpu())

grid_img = make_grid(images, nrow=8)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()
