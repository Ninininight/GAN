#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from utils import LambdaLR, ImagePool
from utils import weights_init_normal, get_config
from datasets import ImageDataset, ValDataset
from CycleGan import *
from utils import Resize, ToTensor, smooothing_loss
from reg import Reg
from torchvision.transforms import RandomAffine
from transformer import Transformer_2D
from skimage.metrics import structural_similarity
import numpy as np
import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class Cyc_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config 
        ## def networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        self.n_epochs = config['n_epochs']
        self.decay_epoch = config['decay_epoch']
        self.lr = config['lr']

        if config['regist']:
            self.R_A = Reg(config['size'], config['size'],config['input_nc'],config['input_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()
            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        if config['bidirect']:
            self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
            self.netD_A = Discriminator(config['input_nc']).cuda()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),lr=config['lr'], betas=(0.5, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        else:
            self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()

        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        # ImagePool for fake images
        self.fake_A_buffer = ImagePool(config['pool_size'])
        self.fake_B_buffer = ImagePool(config['pool_size'])

        #Dataset loader
        level = config['noise_level']    # set noise level

        # 调整 transforms 的顺序
        transforms_1 = [Resize(size_tuple = (config['size'], config['size'])),
                       RandomAffine(degrees=level,translate=[0.02*level, 0.02*level],scale=[1-0.02*level, 1+0.02*level]),
                       ToTensor()]

        # 调整 transforms 的顺序
        transforms_2 = [Resize(size_tuple = (config['size'], config['size'])),
                       RandomAffine(degrees=1,translate=[0.02, 0.02],scale=[0.98, 1.02]),
                       ToTensor()]

        self.dataloader = DataLoader(ImageDataset(config['dataroot'], level, transforms_1=transforms_1, transforms_2=transforms_2, unaligned=False,),
                                 batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'])

        # 调整 transforms 的顺序
        val_transforms = [Resize(size_tuple = (config['size'], config['size'])),
                              ToTensor()]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_ =val_transforms, unaligned=False),
                                 batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        # 初始化 losses 记录
        self.losses = {
            'G': [],
            'D_B': [],
            'D_A': [],
            'GAN_A2B': [],
            'GAN_B2A': [],
            'cycle_ABA': [],
            'cycle_BAB': [],
            'SR': [],
            'SM': []
        }
        self.writer = SummaryWriter(log_dir=os.path.join('runs', self.config['name']))
        

    def update_learning_rate(self, epoch):
        """线性衰减学习率，从 decay_epoch 开始，逐步减到 0"""
        if epoch >= self.decay_epoch:
            decay_factor = (self.n_epochs - epoch) / (self.n_epochs - self.decay_epoch)
            new_lr = self.lr * decay_factor
            for opt in [self.optimizer_G, self.optimizer_D_B] + \
                       ([self.optimizer_D_A] if self.config['bidirect'] else []) + \
                       ([self.optimizer_R_A] if self.config['regist'] else []):
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr
            print(f'[Epoch {epoch}] Learning rate updated to {new_lr:.8f}')
        else:
            # decay_epoch 前保持初始学习率
            for opt in [self.optimizer_G, self.optimizer_D_B] + \
                       ([self.optimizer_D_A] if self.config['bidirect'] else []) + \
                       ([self.optimizer_R_A] if self.config['regist'] else []):
                for param_group in opt.param_groups:
                    param_group['lr'] = self.lr

    def train(self):
        ###### Training ######
        for epoch in range(self.config['epoch'], self.n_epochs):
            for i, batch in enumerate(self.dataloader):
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))

                # Generators A2B and B2A
                self.optimizer_G.zero_grad()

                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                if self.config['bidirect']:
                    fake_A = self.netG_B2A(real_B)
                    pred_fake = self.netD_A(fake_A)
                    loss_GAN_B2A = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)
                else:
                    loss_GAN_B2A = 0

                # Registration loss if enabled
                if self.config['regist']:
                    Trans = self.R_A(fake_B.detach(), real_B)
                    SysRegist_A2B = self.spatial_transform(fake_B, Trans)
                    # 修正：将 real_B 缩放到与 SysRegist_A2B 相同的尺寸
                    real_B_resized = F.interpolate(real_B, size=(SysRegist_A2B.size(2), SysRegist_A2B.size(3)), mode='bilinear', align_corners=False)
                    SR_loss = self.config['Corr_lamda'] * self.L1_loss(SysRegist_A2B, real_B_resized)
                    SM_loss = self.config['Smooth_lamda'] * smooothing_loss(Trans)
                else:
                    SR_loss = 0
                    SM_loss = 0

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B) if self.config['bidirect'] else real_A 
                loss_cycle_ABA = self.config['Cyc_lamda'] * self.L1_loss(recovered_A, real_A)

                recovered_B = self.netG_A2B(fake_A) if self.config['bidirect'] else real_B
                loss_cycle_BAB = self.config['Cyc_lamda'] * self.L1_loss(recovered_B, real_B) if self.config['bidirect'] else 0

                # Total loss for Generators and Registration Network
                loss_G = loss_GAN_A2B + loss_cycle_ABA + loss_GAN_B2A + loss_cycle_BAB + SR_loss + SM_loss
                loss_G.backward()
                self.optimizer_G.step()
                if self.config['regist']:
                    self.optimizer_R_A.step()

                ###################################
                # Discriminator B
                ###################################
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                fake_B = self.fake_B_buffer.query(fake_B.detach())
                pred_fake = self.netD_B(fake_B)
                loss_D_fake = self.MSE_loss(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()
                self.optimizer_D_B.step()

                ###################################
                # Discriminator A (if bidirectional)
                ###################################
                if self.config['bidirect']:
                    self.optimizer_D_A.zero_grad()

                    # Real loss
                    pred_real = self.netD_A(real_A)
                    loss_D_real = self.MSE_loss(pred_real, self.target_real)

                    # Fake loss
                    fake_A = self.fake_A_buffer.query(fake_A.detach())
                    pred_fake = self.netD_A(fake_A)
                    loss_D_fake = self.MSE_loss(pred_fake, self.target_fake)

                    # Total loss
                    loss_D_A = (loss_D_real + loss_D_fake)*0.5
                    loss_D_A.backward()
                    self.optimizer_D_A.step()
                else:
                    loss_D_A = 0

                # Record losses
                self.losses['G'].append(loss_G.item())
                self.losses['D_B'].append(loss_D_B.item())
                if self.config['bidirect']:
                    self.losses['D_A'].append(loss_D_A.item())
                self.losses['GAN_A2B'].append(loss_GAN_A2B.item())
                if self.config['bidirect']:
                    self.losses['GAN_B2A'].append(loss_GAN_B2A.item())
                self.losses['cycle_ABA'].append(loss_cycle_ABA.item())
                if self.config['bidirect']:
                    self.losses['cycle_BAB'].append(loss_cycle_BAB.item() if loss_cycle_BAB != 0 else 0)
                if self.config['regist']:
                    self.losses['SR'].append(SR_loss.item())
                    self.losses['SM'].append(SM_loss.item())
            # 记录每个 epoch 的平均 loss 到 TensorBoard
            for key, values in self.losses.items():
                if len(values) >= len(self.dataloader):
                    # 每个 epoch 的平均值
                    epoch_loss = np.mean(values[-len(self.dataloader):])
                    self.writer.add_scalar(f'Loss/{key}', epoch_loss, epoch)
 
            # 更新学习率
            self.update_learning_rate(epoch)
           
            # Save models checkpoints
            output_dir = os.path.join('output', self.config['name'])
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.netG_A2B.state_dict(), os.path.join(output_dir, 'netG_A2B_%d.pth' % epoch))
            torch.save(self.netD_B.state_dict(), os.path.join(output_dir, 'netD_B_%d.pth' % epoch))
            if self.config['bidirect']:
                torch.save(self.netG_B2A.state_dict(), os.path.join(output_dir, 'netG_B2A_%d.pth' % epoch))
                torch.save(self.netD_A.state_dict(), os.path.join(output_dir, 'netD_A_%d.pth' % epoch))
            if self.config['regist']:
                torch.save(self.R_A.state_dict(), os.path.join(output_dir, 'R_A_%d.pth' % epoch))

            # Run validation
            if (epoch % 5) == 0:
                print('Running Validation...')
                PSNR = 0
                MAE = 0
                SSIM = 0
                num = 0
                for val_i, val_batch in enumerate(self.val_data):
                    val_real_A = Variable(self.input_A.copy_(val_batch['A']))
                    val_real_B = Variable(self.input_B.copy_(val_batch['B']))
                    val_fake_B = self.netG_A2B(val_real_A).data.cpu().numpy()
                    val_real_B_np = val_real_B.data.cpu().numpy()

                    for b in range(val_fake_B.shape[0]):
                        PSNR += self.PSNR(val_fake_B[b,0], val_real_B_np[b,0])
                        MAE += self.MAE(val_fake_B[b,0], val_real_B_np[b,0])
                        # 修改：使用新的函数调用方式
                        SSIM += structural_similarity(val_fake_B[b,0], val_real_B_np[b,0], data_range=2) 
                        num += 1
                print(f"epoch:{epoch}/80")
                print ('MAE:',MAE/num)
                print ('PSNR:',PSNR/num)
                print ('SSIM:',SSIM/num)
                print(" ")
                self.writer.add_scalar("Val/MAE", MAE / num, epoch)
                self.writer.add_scalar("Val/PSNR", PSNR / num, epoch)
                self.writer.add_scalar("Val/SSIM", SSIM / num, epoch)

                
        
        self.writer.close()




    def test(self, epoch, output_dir):
        self.netG_A2B.eval()
        os.makedirs(output_dir, exist_ok=True)

        if self.config['bidirect']:
            self.netG_B2A.eval()

        # Load checkpoint
        ckpt_dir = os.path.join('output', self.config['name'])
        self.netG_A2B.load_state_dict(torch.load(os.path.join(ckpt_dir, f'netG_A2B_{epoch}.pth')))
        if self.config['bidirect']:
            self.netG_B2A.load_state_dict(torch.load(os.path.join(ckpt_dir, f'netG_B2A_{epoch}.pth')))

        if self.config['regist']:
            self.R_A.load_state_dict(torch.load(os.path.join(ckpt_dir, f'R_A_{epoch}.pth')))
            self.spatial_transform.eval()

        PSNR = 0
        MAE = 0
        SSIM = 0
        num = 0

        with torch.no_grad():
            for val_i, val_batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(val_batch['A']))
                real_B = Variable(self.input_B.copy_(val_batch['B']))
                fake_B = self.netG_A2B(real_A)

                # 保存生成图像
                fake_img = fake_B[0].detach().cpu().numpy()  # (C,H,W)
                fake_img = ((fake_img + 1) / 2 * 255).astype(np.uint8)
                fake_img = np.transpose(fake_img, (1, 2, 0))  # HWC
                Image.fromarray(fake_img.squeeze()).save(os.path.join(output_dir, f"fake_B_{val_i}.png"))

                # 计算指标
                val_fake_B = fake_B.data.cpu().numpy()
                val_real_B_np = real_B.data.cpu().numpy()

                for b in range(val_fake_B.shape[0]):
                    fake = val_fake_B[b, 0]
                    real = val_real_B_np[b, 0]

                    PSNR += self.PSNR(fake, real)
                    MAE += self.MAE(fake, real)
                    SSIM += structural_similarity(fake, real, data_range=2)
                    num += 1

        print(f'[Test@Epoch {epoch}] PSNR: {PSNR/num:.3f}, SSIM: {SSIM/num:.3f}, MAE: {MAE/num:.3f}')

    def PSNR(self, fake, real):
        mask = (real != -1).any(axis=0)  # 有效像素掩码（H, W）
        if not np.any(mask):
            return 0

        # 计算所有通道的 MSE
        mse = np.mean(((fake[:, mask] + 1) / 2 - (real[:, mask] + 1) / 2) ** 2)
        if mse < 1e-10:
            return 100
        return 20 * np.log10(1.0 / np.sqrt(mse))


    def MAE(self, fake, real):
        # 输入 fake 和 real 的维度应为 (3, H, W)
        mask = (real != -1).any(axis=0)  # 找出所有通道中有效的像素位置（H, W）
        mae = np.abs(fake[:, mask] - real[:, mask]).mean()  # 计算所有通道的联合误差
        return mae / 2  # 假设输入范围是 [-1, 1]


    def save_deformation(self,defms,root):
        heatmapshow = None
        defms_ = defms.data.cpu().float().numpy()
        dir_x = defms_[0]
        dir_y = defms_[1]
        x_max,x_min = dir_x.max(),dir_x.min()
        y_max,y_min = dir_y.max(),dir_y.min()
        dir_x = ((dir_x-x_min)/(x_max-x_min))*255
        dir_y = ((dir_y-y_min)/(y_max-y_min))*255
