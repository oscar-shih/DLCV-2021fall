import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    #   print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, dir_path):
        self.path = dir_path
        self.data = sorted(os.listdir(dir_path))
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((.5,.5,.5), (.5,.5,.5 ))
        ])
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.path, self.data[index]))), int(self.data[index].split('_')[0])

    def __len__(self):
        return len(self.data)

ts = TrainSet('./generate')

if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "./Classifier.pth"
    load_checkpoint(path, net)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    raw_p = []
    if torch.cuda.is_available():
        net = net.to(device)
    dataloader = torch.utils.data.DataLoader(
        ts,
        batch_size=1000,
        shuffle=False,
    )
    for imgs, labels in dataloader:
        with torch.no_grad():
            logits = net(imgs.to(device))
        predictions = logits.argmax(dim=-1).cpu().numpy()
    labels = labels.to('cpu').numpy()
    print(np.sum(predictions == labels), len(predictions))
