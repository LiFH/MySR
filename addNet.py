#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   addNet.py    
@Contact :   fh.li@foxmail.com
@Modify Time  2020/4/13 8:50 AM  
'''
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
from data_utils import *
from loss_function.loss import GeneratorLoss
# from model.srgan import Generator, Discriminator
from model.msingan import DownGenerator, UpGenerator, WDiscriminator
from utils import *
from model.zssr import ZSSRNet


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=100000, type=int, help='train epoch number')

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
    out_path = 'training_results/SRF_SRFBN' + str(UPSCALE_FACTOR) + '/'
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


def test(model,lr,name='' ,epoch=''):
    net.eval()
    sr = model(lr)
    # print('after:', psnr_ssim(lr, sr))
    sr = sr.squeeze(0)

    save_image(path='results/zssr/'+str(name)+str(epoch)+'.png',img = sr)

def save_image(path,img):
    from torchvision.transforms import  ToPILImage

    img = ToPILImage()(img)
    img.save(path)

from torch.nn import init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # train_set = SingalDataSR2HR('baby')

    # train_set = TrainDatasetFromFolder('data/HR/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    # train_set = TrainDatasetFromFolderUnpairHR('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)

    # val_set = ValDatasetFromFolder('data/HR/Set5', upscale_factor=UPSCALE_FACTOR)
    # train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
    # val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    name = 'woman'
    lr,hr = SingleDataSR2HR(name)
    lr = lr.unsqueeze(0)
    hr = hr.unsqueeze(0)
    test_lr = SingData('data/SR/BI/SRFBN/Set5/HRx2/'+name+'_HR_x2.png')
    test_lr = test_lr.unsqueeze(0)

    start_epoch = 93890
    load_epoch_model = False

    print('origin:', psnr_ssim(lr, hr))
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    # running_results = {'batch_sizes': 0, 'd_loss': 0, 'lr_g_loss': 0, 'd_score': 0, 'lr_g_score': 0, }
    # model

    net = ZSSRNet()
    print('# upsample generator  parameters:', get_network_params(net))
    learning_rate = 1
    # optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # train_bar = tqdm(train_loader)
    net.apply(weights_init_kaiming)
    if(start_epoch > 0):
        load_model(net,'ZSSR_epochs/'+ name+'/net_epoch_%d.pth' % ( start_epoch))

    test(net,test_lr,name,start_epoch)
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        net.train()
        batch_size = lr.size(0)
        sr = net(lr)
        net.zero_grad()

        # loss_fun = nn.L1Loss()
        loss_fun = nn.MSELoss()

        loss = loss_fun(sr, hr)
        loss.backward()

        optimizer.step()

        net.eval()
        print('epoch:', epoch ,'after:',psnr_ssim(sr,hr))

        torch.save(net.state_dict(), 'ZSSR_epochs/'+ name+'/net_epoch_%d.pth' % ( epoch))


        # valG(net)




