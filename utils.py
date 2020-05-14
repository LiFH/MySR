#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py    
@Contact :   fh.li@foxmail.com
@Modify Time  2020/3/30 3:56 PM  
'''
import torch
import pytorch_ssim
from math import log10

def get_network_params(net):
    return  sum(param.numel() for param in net.parameters())




def RGB_normalization(out):
  a = torch.rand((1, 1))
  zero = torch.zeros_like(a)
  one = torch.ones_like(a)
  if torch.cuda.is_available():
    zero = zero.cuda()
    one = one.cuda()
  out = torch.where(out > 1, one, out)
  out = torch.where(out < 0, zero ,out)
  return out


# def init_models(opt):
#
#     #generator initialization:
#     netG = GeneratorConcatSkip2CleanAdd(opt)
#     netG.apply(weights_init)
#     if opt.netG != '':
#         netG.load_state_dict(torch.load(opt.netG, map_location='cpu'))
#     print(netG)
#     # neG = netG.module
#     #discriminator initialization:
#     netD = WDiscriminator(opt)
#     netD.apply(weights_init)
#     if opt.netD != '':
#         netD.load_state_dict(torch.load(opt.netD,map_location='cpu'))
#     print(netD)
#     # netD = netD.module
#
#     return netD, netG


def psnr_ssim(sr, hr, batch_size=1):
  '''
  To calculate the psnr and ssim
  PSNR=10*log10((2^n-1)^2/MSE)
  '''
  valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_size': 0}
  valing_results['batch_size'] = batch_size
  batch_mse = ((sr - hr) ** 2).data.mean()
  valing_results['mse'] += batch_mse * batch_size
  batch_ssim = pytorch_ssim.ssim(sr, hr).item()
  valing_results['ssims'] += batch_ssim * batch_size
  valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_size']))
  valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_size']
  return valing_results['psnr'], valing_results['ssim']