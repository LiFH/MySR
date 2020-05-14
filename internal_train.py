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
from model.msingan import DownGenerator, UpGenerator, WDiscriminator, Upsample
# from utils import *
from model.zssr import ZSSRNet
from utils import util



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


    name = 'bird'
    lr,hr = SingleDataLR2HR(name)
    print(lr.size(),hr.size())
    lr = lr.unsqueeze(0)
    hr = hr.unsqueeze(0)

    test_lr = SingData('../data/SR/BI/SRFBN/Set5/HRx2/'+name+'_HR_x2.png')
    test_lr = test_lr.unsqueeze(0)

    start_epoch = 0
    load_epoch_model = False

    # print('origin:', psnr_ssim(lr, hr))

    # sr = util.RGB_normalization(sr)
    upnet = Upsample(UPSCALE_FACTOR)

    net = UpGenerator(opt.upscale_factor)

    print('# upsample generator  parameters:', util.get_network_params(net))




    up_lr = upnet(lr)


    # utils.save_image(up_lr, 'internal_result/bicubic.png')


    np_lr, np_sr = util.Tensor2np([up_lr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
    psnr, ssim = util.calc_metrics(np_lr, np_sr, crop_border=opt.upscale_factor)
    print('lr,sr :',psnr,ssim)


    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    # learning_rate = 0.1
    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # train_bar = tqdm(train_loader)
    # net.apply(weights_init_kaiming)
    # if(start_epoch > 0):
    #     # load_model(net,'epochs/'+ name+'/net_epoch_%d.pth' % ( start_epoch))
    #     opt.net = 'epochs/netG_Up_epoch_%d_%d.pth' % (UPSCALE_FACTOR, start_epoch)



    model_net = 'epochs/netG_Up_epoch_%d_%d.pth' % (UPSCALE_FACTOR, 72)
    load_model(net, model_net)

    net.eval()
    test_sr = net(up_lr)

    # utils.save_image(test_sr, 'internal_result/origin.png')

    # test_sr2 = net(test_sr)
    #
    # utils.save_image(test_sr2, 'internal_result/origin2.png')

    # print('epoch:', epoch ,'after:',psnr_ssim(sr,hr))
    np_lr, np_sr = util.Tensor2np([test_sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
    psnr, ssim = util.calc_metrics(np_lr, np_sr, crop_border=opt.upscale_factor)
    print('mynet',psnr, ssim)


    # test(net,test_lr,name,start_epoch)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        up_hr = upnet(hr)
        utils.save_image(up_hr, 'internal_result/'+ name +'/bicubic.png')


        net.eval()
        origin_sr = net(up_hr)
        utils.save_image(origin_sr, 'internal_result/'+ name +'/origin_sr.png')



        net.train()
        batch_size = lr.size(0)

        lr2 = upnet(lr)
        sr = net(lr2)
        net.zero_grad()

        loss_fun = nn.L1Loss()
        # loss_fun = nn.MSELoss()

        loss = loss_fun(sr, hr)
        loss.backward()

        optimizer.step()




        # test
        net.eval()
        lr2 = upnet(lr)
        sr = net(lr2)
        np_lr2, np_sr2 = util.Tensor2np([sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
        psnr, ssim = util.calc_metrics(np_lr2, np_sr2, crop_border=opt.upscale_factor)
        print('after',psnr, ssim)
        # print(lr.size())


        ssr = net(up_hr)
        # print(sr2.size())


        ssr = util.RGB_normalization(ssr)
        utils.save_image(ssr, 'internal_result/'+name+ '/' + str(epoch)+'.png')

        # torch.save(net.state_dict(), 'ZSSR_epochs/'+ name+'/net_epoch_%d.pth' % ( epoch))


        # valG(net)




