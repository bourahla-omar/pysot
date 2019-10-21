from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class AlexNetP1(nn.Module):
    configs = [3, 96, 256, 384]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNetP1.configs))
        super(AlexNetP1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.feature_size = configs[3]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class AlexNetP2(nn.Module):
    configs = [484, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNetP2.configs))
        super(AlexNetP2, self).__init__()
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=3),
            nn.BatchNorm2d(configs[1]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=3),
            nn.BatchNorm2d(configs[2]),
            )
        self.feature_size = configs[2]

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer5(x)
        return x


def alexnetp1(**kwargs):
    return AlexNetP1(**kwargs)


def alexnetp2(**kwargs):
    return AlexNetP2(**kwargs)