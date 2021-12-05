import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, label_data=None, transform=None):
        self.transform = transform
        self.label_data = label_data

        self.img_paths = []
        self.labels = []

        if label_data is not None:
            for d in self.label_data:
                img_path, label = d.split(',')
                self.img_paths.append(os.path.join(root, img_path))
                self.labels.append(int(label))
        else:
            for fn in glob.glob(os.path.join(root, '*.png')):
                self.img_paths.append(fn)
                self.labels.append(0)

        self.len = len(self.img_paths)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.img_paths[index], self.labels[index]
        image = Image.open(image_fn).convert('RGB')
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class myDataset_2(Dataset):
    def __init__(self, root, label_data=None, transform=None):
        self.transform = transform
        self.label_data = label_data
        self.images = []
        self.labels = []

        if label_data is not None:
            for d in self.label_data:
                img_path, label = d.split(',')
                image = Image.open(os.path.join(root, img_path)).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                    
                self.images.append(image)
                self.labels.append(int(label))
        else:
            for fn in glob.glob(os.path.join(root, '*.png')):
                image = Image.open(fn).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)

                self.images.append(image)
                self.labels.append(0)

        self.len = len(self.images)


    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image, label = self.images[index], self.labels[index]

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len
