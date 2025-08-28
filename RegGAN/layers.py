from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

scale_eval = Falsesss

alpha = 0.02
beta = 0.00002

resnet_n_blocks = 1

norm_layer = partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
align_corners = False
up_sample_mode = 'bilinear'


def custom_init(m):
    m.data.normal_(0.0, alpha)


def get_init_function(activation, init_function, **kwargs):
    """Get the initialization function from the given name."""
    a = 0.0
    if activation == 'leaky_relu':
        a = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']

    gain = 0.02 if 'gain' not in kwargs else kwargs['gain']
    if isinstance(init_function, str):
        if init_function == 'kaiming':
            activation = 'relu' if activation is None else activation
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation, mode='fan_in')
        elif init_function == 'dirac':
            return torch.nn.init.dirac_
        elif init_function == 'xavier':
            activation = 'relu' if activation is None else activation
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
        elif init_function == 'normal':
            return partial(torch.nn.init.normal_, mean=0.0, std=gain)
        elif init_function == 'orthogonal':
            return partial(torch.nn.init.orthogonal_, gain=gain)
        elif init_function == 'zeros':
            return partial(torch.nn.init.normal_, mean=0.0, std=1e-5)
    elif init_function is None:
        if activation in ['relu', 'leaky_relu']:
            return partial(torch.nn.init.kaiming_normal_, a=a, nonlinearity=activation)
        if activation in ['tanh', 'sigmoid']:
            gain = torch.nn.init.calculate_gain(nonlinearity=activation, param=a)
            return partial(torch.nn.init.xavier_normal_, gain=gain)
    else:
        return init_function


def get_activation(activation, **kwargs):
    """Get the appropriate activation from the given name"""
    if activation == 'relu':
        return nn.ReLU(inplace=False)
    elif activation == 'leaky_relu':
        negative_slope = 0.2 if 'negative_slope' not in kwargs else kwargs['negative_slope']
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        return None


class Conv(torch.nn.Module):
    """Defines a basic convolution layer.
    The general structure is as follow:

    Conv -> Norm (optional) -> Activation -----------> + --> Output
                                     |                ^
                                     |__ResBlcok__| (optional)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, **kwargs):
        super(Conv, self).__init__()
     
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.resnet_block = ResnetTransformer(out_channels, resnet_n_blocks, init_func) if use_resnet else None
        self.norm = norm_layer(out_channels) if use_norm else None
        self.activation = get_activation(activation, **kwargs)
        # Initialize the weights
        init_ = get_init_function(activation, init_func)
        init_(self.conv2d.weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias.data.zero_()
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2d):
            nn.init.normal_(self.norm.weight.data, 0.0, 1.0)
            nn.init.constant_(self.norm.bias.data, 0.0)

    def forward(self, x):
        x = self.conv2d(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.resnet_block is not None:
            x = self.resnet_block(x)
        return x


class UpBlock(torch.nn.Module):
    def __init__(self, nc_down_stream, nc_skip_stream, nc_out, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, refine=False, use_resnet=False, use_add=False, use_attention=False, **kwargs):
        super(UpBlock, self).__init__()
        # 调试: 打印传入 UpBlock 的参数
       

        if 'nc_inner' in kwargs:
            nc_inner = kwargs['nc_inner']
        else:
            nc_inner = nc_out

        self.refine = refine
        self.use_add = use_add

        # Define the layers
        if self.use_add:
            self.upconv = Conv(in_channels=nc_down_stream, out_channels=nc_inner, kernel_size=kernel_size, stride=stride, padding=padding,
                               init_func=init_func, use_norm=use_norm, activation=activation, bias=bias, use_resnet=False)
            self.conv_fuse = Conv(in_channels=nc_skip_stream, out_channels=nc_inner, kernel_size=kernel_size, stride=stride, padding=padding,
                               init_func=init_func, use_norm=use_norm, activation=activation, bias=bias, use_resnet=False)
        else:
            # 修正：in_channels应该是nc_down_stream，因为这是reg.py中计算得到的总通道数。
            # nc_skip_stream是跳跃连接的通道数，不应再次相加。
           
            self.upconv = Conv(in_channels=nc_down_stream, out_channels=nc_inner, kernel_size=kernel_size, stride=stride, padding=padding,
                               init_func=init_func, use_norm=use_norm, activation=activation, bias=bias, use_resnet=use_resnet)

        if self.refine:
            self.refinement = Conv(in_channels=nc_inner, out_channels=nc_out, kernel_size=3, stride=1, padding=1,
                                   init_func=init_func, use_norm=use_norm, activation=activation, bias=bias)

    def forward(self, x, skip_x):
        if self.use_add:
            x = F.interpolate(x, (skip_x.size(2), skip_x.size(3)), mode=up_sample_mode, align_corners=align_corners)
            x = self.upconv(x)
            skip_x = self.conv_fuse(skip_x)
            x += skip_x
        else:
            x = F.interpolate(x, (skip_x.size(2), skip_x.size(3)), mode=up_sample_mode, align_corners=align_corners)
           
            x = torch.cat([x, skip_x], 1)
          
            x = self.upconv(x)

        if self.refine:
            x = self.refinement(x)
        return x


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation='relu',
                 init_func='kaiming', use_norm=False, use_resnet=False, use_pool=False, **kwargs):
        super(DownBlock, self).__init__()
        self.conv = Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         init_func=init_func, use_norm=use_norm, activation=activation, bias=bias, use_resnet=use_resnet)
        self.pool = nn.AvgPool2d(2) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class ResnetTransformer(nn.Module):
    """Resnet-like transformer block."""

    def __init__(self, dim, n_blocks, init_func):
        super(ResnetTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.conv_block = partial(ResnetBlock, dim, init_func)
        self.res_blocks = [self.conv_block() for _ in range(n_blocks)]
        self.res_blocks = nn.Sequential(*self.res_blocks)

    def forward(self, x):
        return self.res_blocks(x)


class ResnetBlock(nn.Module):
    """Define a Resnet block."""

    def __init__(self, dim, init_func, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        """Initialize the Resnet block.
        A resnet block is a convolution block with skip connections
        (input -- + -- output)
                      |__|
        Args:
            dim (int)      -- input and output channels
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer      -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        self.conv_block = nn.Sequential(*conv_block)
        init_ = get_init_function('relu', init_func)
        # 修正: 初始化倒数第二个模块（最后一个卷积层）的权重
        init_(self.conv_block[-2].weight)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out