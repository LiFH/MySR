#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   psnrtest.py    
@Contact :   fh.li@foxmail.com
@Modify Time  2020/4/16 5:55 PM  
'''

from utils import *

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np



lr_filename = '/Users/lifenghai/workspace/sr/pytorch-zssr/results/womanbothBIx2/zssr3110.png'
# lr_filename = '/Users/lifenghai/workspace/sr/pytorch-zssr/data/SR/BI/SRFBN/Set5/x2/woman_SRFBN_x2.png'
sr_filename = '/Users/lifenghai/workspace/sr/pytorch-zssr/data/HR/Set5/woman_GT.png'
# (32.79205969277163, 0.9474092125892639)
# (35.16700566850005, 0.9728933572769165)

# zssr 33.30223821125559
lr = Image.open(lr_filename)
lr = lr.convert('YCbCr')
lr_y, cb, cr = lr.split()
# lr = rgb2ycbcr(lr)
lr_y = ToTensor()(lr_y)

sr = Image.open(sr_filename)
sr = sr.convert('YCbCr')
sr_y, cb, cr = sr.split()
sr_y = ToTensor()(sr_y)

lr_y = lr_y.unsqueeze(0)
sr_y = sr_y.unsqueeze(0)
print(psnr_ssim(lr_y,sr_y))


###########
lr = Image.open(lr_filename)
# lr = rgb2ycbcr(lr)
lr = ToTensor()(lr)

sr = Image.open(sr_filename)
sr = ToTensor()(sr)

lr = lr.unsqueeze(0)
sr = sr.unsqueeze(0)
print(psnr_ssim(lr,sr))