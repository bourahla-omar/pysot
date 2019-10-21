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
    inputc = 384
    kernels = [2, 3, 5]

    def __init__(self):
        inputc = SA.inputc
        channels = list(map(lambda x: inputc * x**2, SA.kernels))
        kernels = SA.kernels
        super(SA, self).__init__()
        #self.stack1 = nn.Sequential(
        #    nn.ConstantPad2d((1, 0, 1, 0), 0),
        #    nn.Conv2d(inputc, channels[0], kernel_size=kernels[0], stride=1, bias=False),
        #    nn.BatchNorm2d(channels[0])
        #)
        self.stack2 = nn.Sequential(
            nn.Conv2d(inputc, channels[1], kernel_size=kernels[1], stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[1])
        )
        #self.stack3 = nn.Sequential(
        #    nn.Conv2d(inputc, channels[2], kernel_size=kernels[2], stride=1, padding=2, bias=False),
        #    nn.BatchNorm2d(channels[2])
        #)
        #length = self.stack1[1].weight.size()[0]
        #weights = torch.eye(length)
        #self.stack1[1].weight.data = weights.reshape(self.stack1[1].weight.size())

        length = self.stack2[0].weight.size()[0]
        weights = torch.eye(length)
        self.stack2[0].weight.data = weights.reshape(self.stack2[0].weight.size())

        #length = self.stack3[0].weight.size()[0]
        #weights = torch.eye(length)
        #self.stack3[0].weight.data = weights.reshape(self.stack3[0].weight.size())

    def forward(self, x, z):
        xr = x.reshape([x.size()[0], x.size()[1], -1])
        zr = z.reshape([z.size()[0], z.size()[1], -1])
        zt = torch.transpose(zr, 1, 2)
        sa0 = torch.matmul(zt, xr)

        x2 = self.stack2(x)
        z2 = self.stack2(z)
        x2r = x2.reshape([x2.size()[0], x2.size()[1], -1])
        z2r = z2.reshape([z2.size()[0], z2.size()[1], -1])
        z2t = torch.transpose(z2r, 1, 2)
        sa2 = torch.matmul(z2t, x2r)

        sa = torch.softmax(torch.mul(sa0, sa2), 1)
        x = torch.cat((x, sa.reshape([sa.size()[0], sa.size()[1], x.size()[2], x.size()[3]])), 1)

        #midp = sa2[int(sa2.size()[0]/2)+13]
        #imtsh = midp.reshape([x.size()[2], x.size()[3]]).detach().cpu().numpy()
        #imgplot = plt.imshow(sa0.squeeze().detach().numpy())
        #imgplot = plt.imshow(sa2.squeeze().detach().numpy())


        return x

    def TRACK(self, x, zbsa):
        stacked1 = self.stack1(x)
        stacked1 = stacked1.reshape([stacked1.size()[0] * stacked1.size()[1], -1])
        stacked2 = self.stack2(x)
        stacked2 = stacked2.reshape([stacked2.size()[0] * stacked2.size()[1], -1])
        stacked3 = self.stack3(x)
        stacked3 = stacked3.reshape([stacked3.size()[0] * stacked3.size()[1], -1])


        return stacked1

def sablock(**kwargs):
    return SA(**kwargs)
