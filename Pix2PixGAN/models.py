import torch
import torch.nn as nn
from collections import OrderedDict
import networks
import util
from base_model import BaseModel

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        # Call parent initialization (sets opt etc.)
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Device set by train.py (we will keep networks on cpu until train moves them),
        # but keep a device attr to be safe (train.py will set model.device = device).
        self.device = getattr(self, 'device', None)

        # Generator
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, self.gpu_ids)

        # Discriminator (conditional -> input_nc + output_nc)
        if self.isTrain:
            # We will not use a Sigmoid at the end of D; loss function will handle it (BCEWithLogits or MSE)
            use_sigmoid = False
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm,
                                          use_sigmoid, opt.init_type, self.gpu_ids)

        # Model names
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # load networks if needed
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # ImagePool removed/disabled for pix2pix (user requested delete). If you REALLY want it back, add logic here.
            self.fake_pool = None

            # Loss functions: use device-safe GANLoss from networks.py
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan)
            self.criterionL1 = torch.nn.L1Loss()

            # Optimizers: use common pix2pix defaults (beta1=0.5)
            beta1 = getattr(opt, 'beta1', 0.5)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

            # schedulers
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.isTrain:
            self.save_network(self.netD, 'D', label, self.gpu_ids)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        # Expect train.py to set model.device before calling set_input
        device = getattr(self, 'device', None)
        if device is None:
            # fallback to cpu if not set
            device = torch.device('cpu')
            self.device = device
        self.real_A = input['A' if AtoB else 'B'].to(device)
        self.real_B = input['B' if AtoB else 'A'].to(device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        # forward
        self.forward()

        # Update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # Update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        # optional: gradient clipping to stabilize training (uncomment if desired)
        # torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=5.0)
        self.optimizer_G.step()

    def backward_D(self):
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # fake_pool removed, so we use fake directly (detached)
        fake_AB_for_D = fake_AB.detach()
        pred_fake = self.netD(fake_AB_for_D)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # GAN loss (want D to predict real on fake_AB)
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        # Combined
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.detach().cpu())
        fake_B = util.tensor2im(self.fake_B.detach().cpu())
        real_B = util.tensor2im(self.real_B.detach().cpu())
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        return ret_visuals

    def get_image_paths(self):
        return self.image_paths

    def get_current_errors(self):
        ret_errors = OrderedDict([
            ('G_GAN', float(self.loss_G_GAN.item())),
            ('G_L1', float(self.loss_G_L1.item())),
            ('D_Real', float(self.loss_D_real.item())),
            ('D_Fake', float(self.loss_D_fake.item()))
        ])
        return ret_errors
 