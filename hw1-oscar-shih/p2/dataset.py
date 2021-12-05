import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

def mask_target(mask_path):
    # Collaborate with b08902134 and b07502071
    mask = Image.open(mask_path)
    mask = np.array(mask).astype(np.uint8)
    mask = (mask >= 128).astype(int)
    mask = 4*mask[:, :, 0] + 2*mask[:, :, 1] + 1*mask[:, :, 2]
    masks = np.zeros((512,512))
    masks[mask == 3] = 0  # (Cyan: 011) Urban land 
    masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    masks[mask == 5] = 2  # (Purple: 101) Rangeland 
    masks[mask == 2] = 3  # (Green: 010) Forest land 
    masks[mask == 1] = 4  # (Blue: 001) Water 
    masks[mask == 7] = 5  # (White: 111) Barren land 
    masks[mask == 0] = 6  # (Black: 000) Unknown
    masks[mask == 4] = 6  # (Red: 100) Unknown
    
    return torch.LongTensor(masks)

class TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        sat_filenames = glob.glob(os.path.join(root, '*.jpg'))
        sat_filenames.sort()
        mask_filenames = glob.glob(os.path.join(root, '*.png'))
        mask_filenames.sort()

        for sat_fn, mask_fn in zip(sat_filenames, mask_filenames):
            self.filenames.append((sat_fn, mask_fn)) 
                
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        sat_fn, mask_fn = self.filenames[index]
        sat = Image.open(sat_fn).convert('RGB')
        if self.transform is not None:
            sat = self.transform(sat)
        return sat, mask_target(mask_fn)

    def __len__(self):
        return len(self.filenames)
