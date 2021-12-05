import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class FCN32s(nn.Module):
    def __init__(self):
        super(FCN32s, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.name = 'FCN32s'
        self.fcn32 = nn.Sequential(
                     nn.Conv2d(512, 4096, 1),
                     nn.ReLU(inplace=True),
                     nn.Dropout2d(),
                     nn.Conv2d(4096, 4096, 1),
                     nn.ReLU(inplace=True),
                     nn.Dropout2d()
                    )
        self.score = nn.Conv2d(4096, 7, 1)
        self.upscore32 =  nn.ConvTranspose2d(7, 7, 64, stride=32, bias=False)
    
    def forward(self, x):
        x_shape = x.shape
        x = self.vgg.features(x)
        x = self.fcn32(x)
        x = self.score(x)
        x = self.upscore32(x)
        x = x[:, :, 32: 32 + x_shape[2], 32: 32 + x_shape[3]]
        return x


class FCN8s(nn.Module):
    def __init__(self):
        super(FCN8s, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.conv6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.conv7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.score = nn.Conv2d(4096, 7, 1)
        self.score_pool3 = nn.Conv2d(256, 7, 1)
        self.score_pool4 = nn.Conv2d(512, 7, 1)
        self.upsample2_1 =  nn.ConvTranspose2d(7, 7, 4, stride=2, bias=False)
        self.upsample2_2 =  nn.ConvTranspose2d(7, 7, 4, stride=2, bias=False)
        self.upsample8 =  nn.ConvTranspose2d(7, 7, 16, stride=8, bias=False)
    
    def forward(self, x):
        x_shape = x.shape
        pool3 = self.vgg.features[:17](x)
        pool4 = self.vgg.features[17:24](pool3)
        x = self.vgg.features[24:](pool4)

        x = self.drop6(self.relu6(self.conv6(x)))
        x = self.drop7(self.relu7(self.conv7(x)))
        x = self.score(x)
        x = self.upsample2_1(x)

        pool4 = self.score_pool4(pool4)
        pool4_shape = pool4.shape

        x = x[:, :, 1: 1 + pool4_shape[2], 1: 1 + pool4_shape[3]] + pool4
        x = self.upsample2_2(x)

        pool3 = self.score_pool3(pool3)
        pool3_shape = pool3.shape

        x = x[:, :, 1: 1 + pool3_shape[2], 1: 1 + pool3_shape[3]] + pool3
        x = self.upsample8(x)
        x = x[:, :, 4: 4 + x_shape[2], 4: 4 + x_shape[3]]
        return x

class DeepLabv3_ResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super(DeepLabv3_ResNet50, self).__init__()
        self.name = "DeepLabv3_Resnet50"
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)['out']
        return x

if __name__ == '__main__':
    m = DeepLabv3_ResNet50(7)
    print(m)