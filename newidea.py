#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   newidea.py.py    
@Contact :   fh.li@foxmail.com
@Modify Time  2020/4/9 10:49 AM  
'''

import argparse
import os
from math import log10
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TrainDatasetFromFolderUnpairHR
from loss_function.loss import GeneratorLoss
# from model.srgan import Generator, Discriminator
from model.msingan import DownGenerator, UpGenerator, WDiscriminator
from utils import *
from model.metasr import MetaRDN

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=600, type=int, help='train epoch number')

from option import args

def save_model(net, name, UPSCALE_FACTOR, epoch):
    torch.save(net.state_dict(), 'epochs/' + name + '_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))

def load_model(net, state_dict):
    net.load_state_dict(torch.load(state_dict,map_location=torch.device('cpu') ))
    return net


def valG(netG_Up,upsampling_net = None):
    netG_Up.eval()
    upsampling_net.eval()
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'lr_g_loss': 0, 'd_score': 0, 'lr_g_score': 0, }
    out_path = 'training_results/SRF_meta' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        for val_lr, val_hr_restore, val_hr in val_bar:
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG_Up(lr)
            sr = upsampling_net(sr)
            sr = RGB_normalization(sr)
            # print(sr)
            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            valing_results['ssims'] += batch_ssim * batch_size
            valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                    valing_results['psnr'], valing_results['ssim']))
            # if netG_Down!=None:
            #     lr2 = valG_Down(netG_Down,lr,sr)

            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0))])
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1
        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (1, index), padding=5)
            index += 1
    running_results['batch_sizes']=5
    # # save model parameters
    # torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # # save loss\scores\psnr\ssim
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results['g_loss'].append(running_results['lr_g_loss'] / running_results['batch_sizes'])
    results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results['g_score'].append(running_results['lr_g_score'] / running_results['batch_sizes'])
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    train_set = TrainDatasetFromFolder('data/HR/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolderUnpairHR('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/HR/Set5', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    start_epoch = 229
    load_epoch_model = True


    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    # running_results = {'batch_sizes': 0, 'd_loss': 0, 'lr_g_loss': 0, 'd_score': 0, 'lr_g_score': 0, }
    # model


    netG = MetaRDN(args)
    print(netG)
    print('# upsample generator  parameters:', get_network_params(netG))
    state_dict = 'save_rdn_model/model_1000.pt'
    netG = load_model(netG, state_dict)
    netG.eval()

    for name, param in netG.named_parameters():
        if param.requires_grad:
            param.requires_grad = False

    #  training
    upsampling_net = UpGenerator(UPSCALE_FACTOR)
    optimizer = optim.Adam(upsampling_net.parameters())
    train_bar = tqdm(train_loader)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        netG.eval()
        upsampling_net.train()
        for lr, hr in train_bar:
            g_update_first = True
            batch_size = lr.size(0)
            for name, param in netG.named_parameters():
                if param.requires_grad:
                    print("requires_grad: True ", name)
            extract_feature = netG(lr)
            # print(extract_feature.shape)
            sr = upsampling_net(extract_feature)
            print(sr)
            # print(hr)
            # print(sr.shape)
            upsampling_net.zero_grad()
            # sr_g_loss = sr_hr_generator_criterion(fake_out, sr_img, hr_img)
            mse_loss = nn.MSELoss()

            sr_g_loss = mse_loss(sr, hr)
            sr_g_loss.backward()

            optimizer.step()

        valG(netG,upsampling_net)



