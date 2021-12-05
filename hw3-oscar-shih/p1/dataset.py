import glob
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = glob.glob(os.path.join(root, '*.jpg'))
        filenames = sorted(filenames)
        for fn in filenames:
            labels = int(fn.split('/')[-1].split('_')[0])
            self.filenames.append((fn, labels)) # (filename, label) pair
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
