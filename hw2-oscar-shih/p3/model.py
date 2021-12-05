import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None



class DANN(nn.Module):
    def __init__(self, code_size=512, n_class=10):
        super(DANN, self).__init__()
        
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