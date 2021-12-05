import torch.nn as nn
from pytorch_pretrained_vit import ViT

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True)
        self.layer = nn.Linear(1000, 37)
    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x
