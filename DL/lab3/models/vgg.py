import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B':
    [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ],
}


def vgg11(num_classes=10):
    return VGG(make_layers(cfg['A']), num_classes=num_classes)


def vgg13(num_classes=10):
    return VGG(make_layers(cfg['B']), num_classes=num_classes)


def vgg16(num_classes=10):
    return VGG(make_layers(cfg['D']), num_classes=num_classes)


def vgg19(num_classes=10):
    return VGG(make_layers(cfg['E']), num_classes=num_classes)
