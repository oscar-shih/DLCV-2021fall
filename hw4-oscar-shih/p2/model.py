import torch
import torch.nn as nn
from torchvision import models
class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.resnet = models.resnet50(pretrained=False).to('cuda')
    #   self.resnet.load_state_dict(torch.load('./improved-net.pt'))
      self.classifier = nn.Linear(1000, 65)

    def forward(self, x):
      x = self.resnet(x)
      x = self.classifier(x)
      return x