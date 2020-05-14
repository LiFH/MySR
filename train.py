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

from model.Resnest import resnest50
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
        loss.backward(retain_graph=True)

        optimizer.step()
        results['loss'] += loss.item() * batch_size
        train_bar.set_description(desc='[%d/%d] Loss: %.4f' % (
            epoch, NUM_EPOCHS, results['loss'] / results['batch_sizes']))


def val(net,netG_Down = None):
    net.eval()
    out_path = 'training_results/SRF_mutil_factor' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
        val_images = []
        batch_psnr1=0
        batch_ssim1 = 0
        for _, _,_,val_lr, val_hr_restore, val_hr in val_bar:
            # name, llr, lhr, lr, hr_restore, hr
            batch_size = val_lr.size(0)
            valing_results['batch_sizes'] += batch_size
            lr = val_lr
            hr = val_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()

            lr = upnet(lr)
            sr = net(lr)
            sr = util.RGB_normalization(sr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            valing_results['mse'] += batch_mse * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            np_lr , np_sr = util.Tensor2np([sr.data[0].float().cpu(), hr.data[0].float().cpu()], 1)
            psnr, ssim = util.calc_metrics(np_lr , np_sr,crop_border=opt.upscale_factor)
            batch_psnr1 += psnr
            batch_ssim1 += ssim
            # print(psnr,ssim)
            # valing_results['ssims'] += batch_ssim * batch_size
            # valing_results['psnr'] = 10 * log10(1 / (valing_results['mse'] / valing_results['batch_sizes']))
            # valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
            val_bar.set_description(
                desc='[converting LR images to SR images] factor: %.2f  PSNR: %.4f dB SSIM: %.4f' % (UPSCALE_FACTOR,
                    batch_psnr1/5, batch_ssim1/5))
            val_images.extend(
                [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                 display_transform()(sr.data.cpu().squeeze(0)),])
        val_images = torch.stack(val_images)
        val_images = torch.chunk(val_images, val_images.size(0) // 15)
        val_save_bar = tqdm(val_images, desc='[saving training results]')
        index = 1

        for image in val_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
            index += 1

    # # save model parameters
    # torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
    # # save loss\scores\psnr\ssim
    # results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    # results['g_loss'].append(running_results['lr_g_loss'] / running_results['batch_sizes'])
    # results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    # results['g_score'].append(running_results['lr_g_score'] / running_results['batch_sizes'])
    # results['psnr'].append(valing_results['psnr'])
    # results['ssim'].append(valing_results['ssim'])
    #
    # if epoch % 10 == 0 and epoch != 0:
    # out_path = 'statistics/'
    # data_frame = pd.DataFrame(
    #     data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
    #           'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
    #     index=range(epoch, epoch + 2))
    # data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results1.csv', index_label='Epoch')


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
    # load_epoch_model =
    opt.net = 'epochs/netG_Up_epoch_%d_%d.pth' % (UPSCALE_FACTOR, start_epoch)


    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    # netG_Up = UpGenerator(opt.upscale_factor)

    upnet = Upsample(UPSCALE_FACTOR)
    net = resnest50()
    print(net)
    print('# upsample generator  parameters:', util.get_network_params(net))

    #

    weights_init(net)
    if start_epoch:
        load_model(net,opt.net)


    # epoch=229
    # valG(netG_Up, netG_Down)

    # loss function


    criterion = nn.L1Loss()
    if torch.cuda.is_available():
        net.cuda()
        criterion.cuda()

    # optimizer
    learning_rate = 0.01
    # optimizer = optim.Adam(netG_Up.parameters())
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


    for epoch in range(start_epoch+1, NUM_EPOCHS + 1):

        running_results = {'batch_sizes': 0, 'd_loss': 0, 'lr_g_loss': 0, 'd_score': 0, 'lr_g_score': 0,}
        print('------------------------')
        for i in range(20,21,1):
            # print("fartor_scale:",i)
            UPSCALE_FACTOR = i/10

            train_set = TrainDatasetFromFolder('../data/HR/DIV2K_train_HR', crop_size=CROP_SIZE,
                                               upscale_factor=UPSCALE_FACTOR)
            # train_set = TrainDatasetFromFolderUnpairHR('data/DIV2K_train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
            val_set = ValDatasetFromFolder('../data/HR/Set5', upscale_factor=UPSCALE_FACTOR)
            test_set = TestDatasetFromFolder('../data/mySet5', upscale_factor = UPSCALE_FACTOR)
            train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
            val_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

            train_bar = tqdm(train_loader)

            train(net, optimizer, criterion, train_bar)
            val(net)
            save_model(net,"net",UPSCALE_FACTOR, epoch)
        # val(netG_Down)
