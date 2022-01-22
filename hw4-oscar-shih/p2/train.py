import random
import torch
import numpy as np
import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

import torch
from byol_pytorch import BYOL
from torchvision import models

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(1126)

device = "cuda" if torch.cuda.is_available() else "cpu"

class OfficeDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.labels = list(sorted(set(pd.read_csv(csv_path).iloc[:,2])))

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.label_dict = {}
        for id, lab in enumerate(self.labels):
          self.label_dict[lab] = id
        
        

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label_str = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        label = self.label_dict[label_str]
        return image, label

    def __len__(self):
        return len(self.data_df)

test_set = OfficeDataset('../hw4_data/office/val.csv', '../hw4_data/office/val')
train_set = OfficeDataset('../hw4_data/office/train.csv', '../hw4_data/office/train')

batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.resnet = models.resnet50(pretrained=False).to('cuda')
      self.resnet.load_state_dict(torch.load('./improved-net.pt'))
      self.classifier = nn.Linear(1000, 65)

    def forward(self, x):
      x = self.resnet(x)
      x = self.classifier(x)
      return x

model = Net().to('cuda')

from transformers import get_cosine_schedule_with_warmup

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train_save(model, epoch, save_interval, log_interval=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    model.train()

    scheduler = get_cosine_schedule_with_warmup(optimizer, 20, 100)
    
    iteration = 0
    for ep in range(epoch):
        scheduler.step()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')
            optimizer.zero_grad()
            # print(data.shape)
            output = model(data).squeeze(0)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # if iteration % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Lr: {}'.format(
                ep, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))
            iteration += 1
        validate(model)
        save_checkpoint('checkpoint/model_%i.pth' % iteration, model, optimizer)

def validate(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to('cuda'), target.to('cuda')
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

train_save(model, 150, 128)
