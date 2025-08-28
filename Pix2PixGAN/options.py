import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='../datasets/bras',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers' )
        self.parser.add_argument('--gpu_ids', type=int, nargs='*', default=[0], help='gpu ids: e.g. 0 1 2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='pix2pix', help='experiment name')
        self.parser.add_argument('--pool_size', type=int, default=50, help='size of image buffer that stores previously generated images')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned', help='dataset loading mode (pix2pix always aligned)')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=4, type=int, help='threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--log_dir', type=str, default='./checkpoints', help='tensorboard logs saved here')
        self.parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--verbose', action='store_true', help='print more debugging information')
        self.parser.add_argument('--serial_batches', action='store_true', help='load datasets in order, otherwise random')
        self.parser.add_argument('--no_lsgan', action='store_true', help='use BCE instead of LSGAN')
        self.parser.add_argument('--lambda_A', type=float, default=15.0, help='weight for L1 loss')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = getattr(self, 'isTrain', False)

        # Convert gpu_ids to device
        if len(self.opt.gpu_ids) == 0 or self.opt.gpu_ids[0] < 0:
            self.opt.gpu_ids = []
        self.opt.device = torch.device(f'cuda:{self.opt.gpu_ids[0]}' if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available() else 'cpu')

        # Automatically compute total epoch
        self.opt.total_epochs = getattr(self.opt, 'niter', 0) + getattr(self.opt, 'niter_decay', 0)

        # Add log_dir for TensorBoard
        self.opt.log_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'logs')
        os.makedirs(self.opt.log_dir, exist_ok=True)

        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=30, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='starting epoch count')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: lambda|step|plateau|cosine')
        self.isTrain = True


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='save results here')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--model_path', default='./checkpoints/experiment_name/latest_net_G.pth', help='specific model path')
        self.isTrain = False
