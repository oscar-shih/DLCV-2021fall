import torch.nn as nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.wide_resnet50_2(pretrained=True, progress=True)
        self.layer = nn.Linear(1000, 50)


    def forward(self, x):
        x = self.model(x)
        x = self.layer(x)
        return x

if __name__ == '__main__':
    m = Model()
    print(m)