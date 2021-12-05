# import os
# import argparse
# import glob
# from dataset import myDataset
# import torch
# from torch.backends import cudnn
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader
# from model import DANN
# from PIL import Image


# def main(config):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
#     ])
#     use_cuda = torch.cuda.is_available()
#     device = torch.device('cuda' if use_cuda else 'cpu')

#     model = DANN().cuda()

#     state = torch.load(config.ckp_path)
#     model.load_state_dict(state['state_dict'])

#     filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
#     filenames = sorted(filenames)

#     out_filename = config.save_path
#     os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

#     model.eval()
#     with open(out_filename, 'w') as out_file:
#         out_file.write('image_name,label\n')
#         with torch.no_grad():
#             for fn in filenames:
#                 data = Image.open(fn).convert('RGB')
#                 data = transform(data)
#                 data = torch.unsqueeze(data, 0)
#                 data = data.cuda()
#                 output, _ = model(data, 1)
#                 pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#                 out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()

#     # Training configuration.
#     parser.add_argument('--img_dir', type=str, default='../hw2_data/digits/svhn/test')
#     parser.add_argument('--target', type=str, default='ckpt/test')
#     parser.add_argument('--ckp_path', default='ckpt/test/3750-dann.pth', type=str, help='Checkpoint path.')
#     args = parser.parse_args()
#     target_dataset_name = args.target
#     model_name = { "usps":"mnistm_usps_adaptation.pth", "mnistm":"svhn_mnistm_adaptation.pth", "svhn": "usps_svhn_adaptation.pth"}  


#     config = parser.parse_args()
#     print(config)
#     main(config)


import random
import os
import sys
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import argparse

model_dir = './p3_model'

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CNNModel(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(CNNModel, self).__init__()
        
        self.feature_extractor_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(True)
        )

        self.feature_extractor_fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, code_size),
            nn.BatchNorm1d(code_size),
            nn.Dropout(),
            nn.ReLU(True)
        )
        
        self.class_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, n_class),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(code_size, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def encode(self, x):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)

        return feature

    def forward(self, x, alpha=1.0):
        feature = self.feature_extractor_conv(x)
        feature = feature.view(-1, 50 * 4 * 4)
        feature = self.feature_extractor_fc(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output

class TestSet(Dataset):
    def __init__(self, path, name):
        self.path = path
        self.data = sorted(os.listdir(self.path))
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        
    def __getitem__(self, index):
        return self.transform(Image.open(os.path.join(self.path, self.data[index])).convert('RGB')), self.data[index]

    def __len__(self):
        return len(self.data)

cuda = True
batch_size = 128
image_size = 28

same_seeds(1126)

parser = argparse.ArgumentParser()
parser.add_argument("--target", default = "mnistm", type = str)
parser.add_argument("--data_dir", default = "./hw2_data/digits/", type = str)
parser.add_argument("--out_csv", default = "./pred.csv", type = str)
args = parser.parse_args()

target_dataset_name = args.target 
model_name = { "usps":"mnistm-usps.pth", "mnistm":"svhn-mnistm.pth", "svhn": "usps-svhn.pth"}


def test():
    target_dataset = TestSet(args.data_dir, target_dataset_name)
    dataloader = DataLoader(target_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)

    model = CNNModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name[args.target]))['state_dict'])
    print('load', os.path.join(model_dir, model_name[args.target]))

    if cuda:
        model = model.cuda()
    model.eval()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    prediction, labels = [], []
    i = 0
    while i < len_dataloader:
        data_target = data_target_iter.next()
        t_img, t_label = data_target
        if cuda:
            t_img = t_img.cuda()

        class_output, _ = model(t_img, alpha=1.0)
        pred = class_output.data.max(1, keepdim=True)[1]
        prediction.extend(pred.cpu().numpy().tolist())
        labels.extend(t_label)
        i += 1

    with open(args.out_csv, 'w') as f:
        f.write('image_name,label\n')
        for lbl, n in zip(labels, prediction):
            f.write('{},{}\n'.format(lbl, n[0]))

test()