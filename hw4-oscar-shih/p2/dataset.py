import glob
import os
from torch.utils.data import Dataset
from PIL import Image

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.filenames = []
        filenames = sorted(glob.glob(os.path.join(root, '*.jpg')))
        for fn in filenames:
            labels = str(fn.split('0000')[0])
            # print(labels)
            self.filenames.append((fn, labels))

    def __getitem__(self, index):
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.filenames)
        