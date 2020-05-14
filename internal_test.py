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
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TestDatasetFromFolder
from loss_function.loss import GeneratorLoss
# from model.srgan import Generator, Discriminator
from model.msingan import DownGenerator, UpGenerator, WDiscriminator, Upsample, weights_init
from utils import util


from torch import nn


parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=600, type=int, help='train epoch number')



def train(net, optimizer,criterion, train_bar):

    net.train()
    results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0,'loss':0}

    for lr, hr in train_bar:
        batch_size = lr.size(0)
        results['batch_sizes'] += batch_size

        lr = Variable(lr)
        hr = Variable(hr)

        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()


        lr = upnet(lr)
        sr = net(lr)


        net.zero_grad()
        loss = criterion(sr,hr)
        print(loss)
        loss.backward(retain_graph=True)

        optimizer.step()
        #
        results['loss'] += loss.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss: %.4f' % (
            epoch, NUM_EPOCHS, results['loss'] / results['batch_sizes']))


def val(net, optimizer,criterion, val_loader, scale_factor):
    # , optimizer,criterion, train_bar
    # net.eval()
    out_path = 'training_results/SRF_mutil_factor' + str(UPSCALE_FACTOR) + '/'


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # with torch.no_grad():
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    batch_psnr=0
    batch_ssim = 0
    for name, llr,lhr, lr, hr_restore, hr in val_bar:
        # print(scale_factor,name[0])
        batch_size = lr.size(0)
        valing_results['batch_sizes'] += batch_size
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()

        llr = Variable(llr)
        lhr = Variable(lhr)
        lr = Variable(lr)
        hr = Variable(hr)

        net.eval()
        __lr = upnet(lr)
        __sr = net(__lr)
        __sr = util.RGB_normalization(__sr)
        np_lr , np_sr = util.Tensor2np([__sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
        psnr, ssim = util.calc_metrics(np_lr , np_sr,crop_border=opt.upscale_factor)
        utils.save_image(__sr, 'internal_result/lr_zero/' + str(scale_factor) + '00' + name[0])

        # print(psnr,ssim)

        # meta-train
        for i in range(0, 0):
            net.train()
            # upnet.train()
            _llr = upnet(llr)

            _sr = net(_llr)
            # print(train_sr)
            net.zero_grad()

            loss = criterion(_sr, lhr)
            # print(loss)
            loss.backward(retain_graph=True)

            optimizer.step()

            net.eval()
            __lr = upnet(lr)
            __sr = net(__lr)
            __sr = util.RGB_normalization(__sr)
            np_lr, np_sr = util.Tensor2np([__sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
            psnr, ssim = util.calc_metrics(np_lr, np_sr, crop_border=opt.upscale_factor)
            print(psnr, ssim)
            # utils.save_image(__sr, 'internal_result/lr_zero/' + str(scale_factor)+str(i) + name[0])

        net.eval()


        # test
        __lr = upnet(lr)
        __sr = net(__lr)
        __sr = util.RGB_normalization(__sr)

        utils.save_image(__sr, 'internal_result/lr_zero/'+str(scale_factor)+name[0])

        np_lr , np_sr = util.Tensor2np([__sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
        psnr, ssim = util.calc_metrics(np_lr , np_sr,crop_border=opt.upscale_factor)
        # print(psnr,ssim)
        batch_psnr += psnr
        batch_ssim += ssim
        val_bar.set_description(
            desc='[converting LR images to SR images] factor: %.2f  PSNR: %.4f dB SSIM: %.4f' % (UPSCALE_FACTOR,
                batch_psnr/5, batch_ssim/5))


def valG_Down(netG,lr,sr):
    netG.eval()

    lr2 = netG(sr)
    lr2 = util.RGB_normalization(lr2)

    # batch_mse = ((lr - lr2) ** 2).data.mean()
    # valing_results['mse'] += batch_mse * batch_size
    # batch_ssim = pytorch_ssim.ssim(sr, hr).item()
    # valing_results['ssims'] += batch_ssim * batch_size
    # valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
    # valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
    # val_bar.set_description(
    #     desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
    #         valing_results['psnr'], valing_results['ssim']))
    return lr2


def save_model(net,name,UPSCALE_FACTOR, epoch):
    torch.save(net.state_dict(), 'epochs/'+name+'_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))


def load_model(net,state_dict):
    print("load model=>")
    net.load_state_dict(torch.load(state_dict))
    return net

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs



    start_epoch = 0

    # optimizer = optim.SGD(netG_Up.parameters(), lr=learning_rate, momentum=0.9)


    for epoch in range(start_epoch+1, NUM_EPOCHS + 1):

        running_results = {'batch_sizes': 0, 'd_loss': 0, 'lr_g_loss': 0, 'd_score': 0, 'lr_g_score': 0,}
        for i in range(11,41,1):

            # load_epoch_model =
            opt.net = 'epochs/netG_Up_epoch_%d_%d.pth' % (UPSCALE_FACTOR, start_epoch)

            results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

            netG_Up = UpGenerator(opt.upscale_factor)
            print('# upsample generator  parameters:', util.get_network_params(netG_Up))

            weights_init(netG_Up)
            if start_epoch:
                load_model(netG_Up, opt.net)

            # loss function

            criterion = nn.L1Loss()
            # criterion = nn.MSELoss()

            if torch.cuda.is_available():
                netG_Up.cuda()
                criterion.cuda()

            # optimizer
            learning_rate = 0.001
            optimizer = optim.Adam(netG_Up.parameters(), lr=learning_rate)
            # optimizer = optim.SGD(netG_Up.parameters(), lr=learning_rate)
            # print("fartor_scale:",i)
            UPSCALE_FACTOR = i/10
            print(UPSCALE_FACTOR)
            upnet = Upsample(UPSCALE_FACTOR)
            train_set = TrainDatasetFromFolder('../data/HR/DIV2K_train_HR', crop_size=CROP_SIZE,
                                               upscale_factor=UPSCALE_FACTOR)
            # train_set = TrainDatasetFromFolderUnpairHR('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
            val_set = ValDatasetFromFolder('../data/HR/Set5', upscale_factor=UPSCALE_FACTOR)
            test_set = TestDatasetFromFolder('../data/mySet5', upscale_factor = UPSCALE_FACTOR)
            train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
            val_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

            # train_bar = tqdm(train_loader)
            # train(netG_Up, optimizer,criterion,
            #       train_bar)
            # #
            val(netG_Up, optimizer,criterion, val_loader, UPSCALE_FACTOR)
            # save_model(netG_Up,"netG_Up",UPSCALE_FACTOR, epoch)
        # val(netG_Down)
