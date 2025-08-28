import random
import time
import datetime
import torch
import torch.nn.functional as F
from visdom import Visdom
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor as ToTensor_
from torchvision.transforms import Compose, Resize as Resize_
from torch.autograd import Variable


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Logger():
    def __init__(self, exp_name, port, n_epochs, batches_epoch):
        self.port = port
        self.viz = Visdom(port=port)
        self.exp_name = exp_name
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_losses = {}
        
    def log(self, losses=None, images=None):
        if losses is not None:
            for loss_name, loss_value in losses.items():
                if loss_name not in self.mean_losses:
                    self.mean_losses[loss_name] = 0
                self.mean_losses[loss_name] += loss_value

            if (self.batch % self.batches_epoch) == 0:
                self.viz.line(
                    X=np.arange(self.epoch, self.epoch+1),
                    Y=np.array([self.mean_losses[loss_name] / self.batches_epoch for loss_name in self.mean_losses]),
                    win=self.exp_name + '_losses',
                    opts=dict(title=self.exp_name + ' losses', legend=list(self.mean_losses.keys())),
                    update='append' if self.epoch > 1 else None
                )
                self.epoch += 1
                self.batch = 1
                self.mean_losses = {}
            else:
                self.batch += 1

        if images is not None:
            for image_name, image_value in images.items():
                self.viz.image(
                    image_value.data.cpu().numpy()[0, ...],
                    win=self.exp_name + '_' + image_name,
                    opts=dict(title=self.exp_name + ' ' + image_name)
                )


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = Variable(images.data.clone())
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images = torch.cat((return_images, image), 0)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images = torch.cat((return_images, tmp), 0)
                else:
                    return_images = torch.cat((return_images, image), 0)
        return_images = return_images[images.size(0):]
        return return_images


def get_config():
    # 这里我们只返回 RegGAN 的配置，与之前修改 train.py 的逻辑保持一致
    config = {
        'name': 'CycleGan',
        'dataroot': './dataset/bras/',
        'val_dataroot': './dataset/bras/',
        'cuda': True,
        'epoch': 0,
        'n_epochs': 200,
        'batchSize': 1,
        'lr': 0.0002,
        'pool_size': 50,
        'size': 256,
        'input_nc': 3,
        'output_nc': 3,
        'Adv_lamda': 1,
        'Cyc_lamda': 10,
        'Idt_lamda': 0.5,
        'Corr_lamda': 1,
        'Smooth_lamda': 1,
        'port': 6019,
        'bidirect': False,
        'regist': True,
        'noise_level': 1
    }
    return config


class ToTensor(object):
    def __call__(self, pic):
        return ToTensor_()(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Resize(object):
    def __init__(self, size_tuple):
        self.size_tuple = size_tuple

    def __call__(self, img):
        if isinstance(img, Image.Image):
            # 如果输入是 PIL.Image 对象，使用 .size
            width, height = img.size
            if self.size_tuple[0] == width and self.size_tuple[1] == height:
                return img
            else:
                return img.resize(self.size_tuple)
        elif isinstance(img, torch.Tensor):
            # 如果输入是 Tensor 对象，使用 .shape
            if len(img.shape) == 2:  # 灰度图 (H, W)
                img = img.unsqueeze(0)
            H, W = img.shape[1], img.shape[2]
            if H == self.size_tuple[0] and W == self.size_tuple[1]:
                return img
            else:
                return F.interpolate(img.unsqueeze(0), size=self.size_tuple, mode='bilinear', align_corners=False).squeeze(0)
        else:
            raise TypeError('Input should be a PIL Image or a torch.Tensor. Got {}'.format(type(img)))

    def __repr__(self):
        return self.__class__.__name__ + '(size_tuple={})'.format(self.size_tuple)


def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d / 2.0
    return grad

def tensor2image(tensor):
    image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3,1,1))
    return image.astype(np.uint8)