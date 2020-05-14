from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import math
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


# def calculate_valid_crop_size(crop_size, upscale_factor):
#     return crop_size - (crop_size % (upscale_factor*10)/10)

def calculate_valid_crop_size(crop_size, upscale_factor):
    # return crop_size - (crop_size % (upscale_factor*10)/10)
    # return int(crop_size/upscale_factor//10* (upscale_factor*10))
    return int(upscale_factor * int(np.floor(crop_size / upscale_factor)))


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(int(math.ceil(crop_size / upscale_factor)), interpolation=Image.BICUBIC),
        ToTensor()
    ])

def lr2bicubic_transform(crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size , interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])



class TrainDatasetFromFolderUnpairHR(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        import random
        random_index = random.sample(range(0, 800), 1)
        unpair_hr_image = self.hr_transform(Image.open(self.image_filenames[(index+random_index)%800]))
        return lr_image, hr_image, unpair_hr_image

    def __len__(self):
        return len(self.image_filenames)

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)

        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class TrainDataset_lr_bicubic_hr(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.bicubic_transform= lr2bicubic_transform(crop_size)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        bicubic_image = self.bicubic_transform(lr_image)
        return lr_image, bicubic_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        print(crop_size)
        print(int(crop_size/self.upscale_factor))
        # print(crop_size,int(crop_size // self.upscale_factor))
        lr_scale = Resize(int(crop_size / self.upscale_factor), interpolation=Image.BICUBIC)
        hr_scale = Resize(int(crop_size), interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.llr_path = dataset_dir + '/LLR/x' + str(upscale_factor)+ '/LLR'
        self.lhr_path = dataset_dir + '/LLR/x' + str(upscale_factor)+ '/HR'
        self.lr_path = dataset_dir + '/LR/x' + str(upscale_factor)
        self.hr_path = dataset_dir + '/HR/x' + str(upscale_factor)
        self.restore_path = dataset_dir + '/Bic/x' + str(upscale_factor)
        self.upscale_factor = upscale_factor

        self.llr_filenames = [join(self.llr_path, x) for x in listdir(self.llr_path) if is_image_file(x)]
        self.lhr_filenames = [join(self.lhr_path, x) for x in listdir(self.lhr_path) if is_image_file(x)]

        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
        self.restore_filenames = [join(self.restore_path, x) for x in listdir(self.restore_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        llr_image = Image.open(self.llr_filenames[index])
        lhr_image = Image.open(self.lhr_filenames[index])

        lr_image = Image.open(self.lr_filenames[index])
        # w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])

        hr_restore_image = Image.open(self.restore_filenames[index])
        # hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        # hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(llr_image),ToTensor()(lhr_image), ToTensor()(lr_image), ToTensor()(hr_restore_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)




import torchvision.transforms.functional as tf
from torchvision import transforms
import random
def my_transform1(image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    mask = mask.rotate(angle)

    image = tf.to_tensor(image)
    mask = tf.to_tensor(mask)
    return image, mask


def my_transform2(image, mask):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    image = tf.to_tensor(image)
    mask = tf.to_tensor(mask)
    return image, mask


class SingalDataSR2HR(Dataset):
    def __init__(self, filename):
        super(SingalDataSR2HR, self).__init__()

        self.sr_filename = 'data/SR/BI/SRFBN/Set5/x2/' + filename +'.png'
        self.hr_filename = 'data/HR/Set5/' + filename + '_GT.png'

    def __getitem__(self, index):
        sr_image = Image.open(self.sr_filename)

        hr_image = Image.open(self.hr_filename)

        sr_tensor, hr_tensor = my_transform1(sr_image, hr_image)
        return sr_tensor, hr_tensor

    def __len__(self):
        return 1




def SingleDataLR2HR(filename):
    # lr_filename = '../data/LR/Set5/LR/x2/' + filename + '_GT.png'
    # hr_filename = '../data/HR/Set5/' + filename + '_GT.png'
    lr_filename = '../data/LR/Set5/LR/x2/' + filename + '_GT.png'
    hr_filename = '../data/HR/Set5/' + filename + '_GT.png'
    lr_image = Image.open(lr_filename)

    hr_image = Image.open(hr_filename)
    return ToTensor()(lr_image), ToTensor()(hr_image)


def SingleDataLR2HRFactor(filename,factor):
    # lr_filename = '../data/LR/Set5/LR/x2/' + filename + '_GT.png'
    # hr_filename = '../data/HR/Set5/' + filename + '_GT.png'
    lr_filename = '../data/LR/Set5/LR/x2/' + filename + '_GT.png'
    hr_filename = '../data/HR/Set5/' + filename + '_GT.png'
    lr_image = Image.open(lr_filename)

    hr_image = Image.open(hr_filename)
    return ToTensor()(lr_image), ToTensor()(hr_image)


def SingData(filename):
    image = Image.open(filename)
    return ToTensor()(image)


