import os
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
# local
from layers import UpBlock, DownBlock, Conv, get_init_function, ResnetTransformer

sampling_align_corners = False


# The number of filters in each block of the encoding part (down-sampling).
ndf = {'A': [32, 64, 64, 128, 64, 64, 64], }
# The number of filters in each block of the decoding part (up-sampling).
nuf = {'A': [64, 64, 64, 64, 64, 64, 32], }
# Indicate if res-blocks are used in the down-sampling path.
use_down_resblocks = {'A': True, }
# indicate the number of res-blocks applied on the encoded features.
resnet_nblocks = {'A': 3, }
# Indicate if the a final refinement layer is applied on the before deriving the deformation field
refine_output = {'A': True, }
# The activation used in the down-sampling path.
down_activation = {'A': 'leaky_relu', }
# The activation used in the up-sampling path.
up_activation = {'A': 'leaky_relu', }


class ResUnet(torch.nn.Module):
    def __init__(self, nc_a, nc_b, cfg, init_func, init_to_identity):
        super(ResUnet, self).__init__()
        act = down_activation[cfg]
        # ------------ Down-sampling path
        self.ndown_blocks = len(ndf[cfg])
        self.nup_blocks = len(nuf[cfg])
        assert self.ndown_blocks >= self.nup_blocks
        in_nf = nc_a + nc_b
        conv_num = 1
        skip_nf = {}
        for out_nf in ndf[cfg]:
           
            setattr(self, 'down_{}'.format(conv_num),
                    DownBlock(in_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                              use_resnet=use_down_resblocks[cfg], use_pool=True))
            skip_nf[conv_num] = out_nf
            in_nf = out_nf
            conv_num += 1

        self.encode_features = ResnetTransformer(in_nf, resnet_nblocks[cfg], init_func)

        # ------------ Up-sampling path
        conv_num -= 1
        for out_nf in nuf[cfg]:
            if conv_num > 0:
                # 打印传入 UpBlock 的参数
                in_channels = in_nf + skip_nf[conv_num]
                
                setattr(self, 'up_{}'.format(conv_num),
                        UpBlock(in_channels, skip_nf[conv_num], out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True,
                                use_resnet=True))
            in_nf = out_nf
            conv_num -= 1

        if 'out_nf' not in locals():
            out_nf = in_nf

        if refine_output[cfg]:
            self.refine = Conv(out_nf, out_nf, 3, 1, 1, activation=act, init_func=init_func, bias=True)
            self.output = Conv(out_nf, 2, 3, 1, 1, activation=None, init_func='zeros', bias=True)  # init with zeros
            if init_to_identity:
                self.output.conv2d.weight.data.zero_()
                self.output.conv2d.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
        else:
            self.refine = None
            self.output = Conv(out_nf, 2, 3, 1, 1, activation=None, init_func='zeros', bias=True)  # init with zeros
            if init_to_identity:
                self.output.conv2d.weight.data.zero_()
                self.output.conv2d.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))

    def forward(self, x_a, x_b):
        x = torch.cat([x_a, x_b], 1)
        skip_vals = {}
        conv_num = 1
        for _ in range(self.ndown_blocks):
            x = getattr(self, 'down_{}'.format(conv_num))(x)
            skip_vals['down_{}'.format(conv_num)] = x
      
            conv_num += 1

        x = self.encode_features(x)
       

        conv_num -= 1
        for _ in range(self.nup_blocks):
            if conv_num > 0:
                s = skip_vals['down_{}'.format(conv_num)]
                x = F.interpolate(x, (s.size(2), s.size(3)), mode='bilinear', align_corners=sampling_align_corners)
              
                x = getattr(self, 'up_{}'.format(conv_num))(x, s)
            conv_num -= 1
        x = self.refine(x)
        x = self.output(x)
        return x

class Reg(nn.Module):
    def __init__(self,height,width,in_channels_a,in_channels_b):
        super(Reg, self).__init__()
        init_func = 'kaiming'
        init_to_identity = True

        self.oh, self.ow = height, width
        self.in_channels_a = in_channels_a
        self.in_channels_b = in_channels_b
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.offset_map = ResUnet(self.in_channels_a, self.in_channels_b, cfg='A', init_func=init_func, init_to_identity=init_to_identity).to(
            self.device)
        self.identity_grid = self.get_identity_grid()

    def get_identity_grid(self):
        x = torch.linspace(-1.0, 1.0, self.ow).view(1, 1, self.ow, 1).expand(1, 1, self.ow, self.oh)
        y = torch.linspace(-1.0, 1.0, self.oh).view(1, 1, 1, self.oh).expand(1, 1, self.ow, self.oh)
        identity_grid = torch.cat([x, y], 1).to(self.device)
        return identity_grid

    def forward(self, x_a, x_b):
        offset = self.offset_map(x_a, x_b)  # Nx2xHxW
        return offset