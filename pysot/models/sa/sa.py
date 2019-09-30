from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt

class SA(nn.Module):
    inputc = 256
    kernels = [2, 3, 5]

    def __init__(self):
        inputc = SA.inputc
        channels = list(map(lambda x: inputc * x**2, SA.kernels))
        kernels = SA.kernels
        super(SA, self).__init__()
        self.stack1 = nn.Sequential(
            nn.ConstantPad2d((1, 0, 1, 0), 0),
            nn.Conv2d(inputc, channels[0], kernel_size=kernels[0], stride=1, bias=False),
            nn.BatchNorm2d(channels[0])
        )
        self.stack2 = nn.Sequential(
            nn.Conv2d(inputc, channels[1], kernel_size=kernels[1], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[1])
        )
        self.stack3 = nn.Sequential(
            nn.Conv2d(inputc, channels[2], kernel_size=kernels[2], stride=1, padding=2, bias=False),
            nn.BatchNorm2d(channels[2])
        )
        length = self.stack1[1].weight.size()[0]
        weights = torch.eye(length)
        self.stack1[1].weight.data = weights.reshape(self.stack1[1].weight.size())

        length = self.stack2[0].weight.size()[0]
        weights = torch.eye(length)
        self.stack2[0].weight.data = weights.reshape(self.stack2[0].weight.size())

        length = self.stack3[0].weight.size()[0]
        weights = torch.eye(length)
        self.stack3[0].weight.data = weights.reshape(self.stack3[0].weight.size())

    def forward(self, x):
        stacked0 = x
        stacked0r = stacked0.reshape([stacked0.size()[0] * stacked0.size()[1], -1])
        sa0 = stacked0r.t().mm(stacked0r)
        #midp = sa0[int(sa0.size()[0]/2)+13]
        #imtsh = midp.reshape([x.size()[2], x.size()[3]]).detach().cpu().numpy()
        #imgplot = plt.imshow(imtsh)

        stacked1 = self.stack1(x)
        stacked1r = stacked1.reshape([stacked1.size()[0] * stacked1.size()[1], -1])
        sa1 = stacked1r.t().mm(stacked1r)
        #midp = sa1[int(sa1.size()[0]/2)+13]
        #imtsh = midp.reshape([x.size()[2], x.size()[3]]).detach().cpu().numpy()
        #imgplot = plt.imshow(imtsh)

        #stacked2 = self.stack2(x)
        #stacked2r = stacked2.reshape([stacked2.size()[0] * stacked2.size()[1], -1])
        #sa2 = (stacked2r.t().mm(stacked2r))
        #midp = sa2[int(sa2.size()[0]/2)+13]
        #imtsh = midp.reshape([x.size()[2], x.size()[3]]).detach().cpu().numpy()
        #imgplot = plt.imshow(imtsh)

        #stacked3 = self.stack3(x)
        #stacked3r = stacked3.reshape([stacked3.size()[0] * stacked3.size()[1], -1])
        #sa3 = (stacked3r.t().mm(stacked3r))
        #midp = sa3[int(sa3.size()[0]/2)+13]
        #imtsh = midp.reshape([x.size()[2], x.size()[3]]).detach().cpu().numpy()
        #imgplot = plt.imshow(imtsh)


        return x

    def TRACK(self, x):
        stacked1 = self.stack1(x)
        stacked1 = stacked1.reshape([stacked1.size()[0] * stacked1.size()[1], -1])
        stacked2 = self.stack2(x)
        stacked2 = stacked2.reshape([stacked2.size()[0] * stacked2.size()[1], -1])
        stacked3 = self.stack3(x)
        stacked3 = stacked3.reshape([stacked3.size()[0] * stacked3.size()[1], -1])


        return stacked1

def sablock(**kwargs):
    return SA(**kwargs)
