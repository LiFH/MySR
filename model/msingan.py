#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   msingan.py    
@Contact :   fh.li@foxmail.com
@Modify Time  2020/3/30 11:02 AM  
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.common import *

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.head = ConvBlock(3, 32, 3, 1, 1)
        self.body = nn.Sequential()
        for i in range(1,3):

            block = ConvBlock(32, 32, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = torch.sigmoid(x)

        return x


class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.head = ConvBlock(3, 32, 3, 1, 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(1,3):
            block = ConvBlock(32, 32, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # ind = int((y.shape[2]-x.shape[2])/2)
        # y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x


class Upsample(nn.Module):
    def __init__(self,scale):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale,
                                    mode='bicubic', align_corners=True)

    def forward(self,x):
        return self.up(x)



class UpGenerator(nn.Module):
    def __init__(self, scale):
        super(UpGenerator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale,
                                    mode='bicubic', align_corners=False)
        self.head = ConvBlock(3, 64, 3, 1, 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(1,7):
            block = ConvBlock(64, 64, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

    def forward(self, x):
        # upsample x to target sr size
        # print('UP')
        # x = self.upsample(x)
        residual = x
        # x = self.upsample(x)
        # y = x
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # x = self.add_mean(x)
        # sr
        return x + residual


class DownGenerator(nn.Module):
    def __init__(self,opt):
        super(DownGenerator, self).__init__()
        # print('down')
        self.opt = opt
        self.scale = [2]
        self.phase = 2
        # n_blocks = opt.n_blocks
        # n_feats = opt.n_feats
        # kernel_size = 3
        n_feats = 3
        # self.down = [
        #     common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
        #                      ) for p in range(self.phase)
        # ]
        # self.down = [
        #     DownBlock(opt,2, 3 ,3 ,3)
        # ]
        #
        # # def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        #
        # self.down = nn.ModuleList(self.down)
        self.down = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.head = ConvBlock(3, 32, 3, 1, 1)  # GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(1,3):
            block = ConvBlock(32, 32, 3, 1, 1)
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
        # sr to lr
        # print('DOWN')
        x = self.down(x)
        x = self.down(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        return x

