import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, normalize=None):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)

        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        if self.normalize:
            x = self.normalize(x)
        return F.elu(x, inplace=True)


class ConvBlock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, normalize=None):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1, normalize=normalize)
        self.conv2 = Conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class MaxPool(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class ResConv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride, normalize=None):
        super(ResConv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = Conv(num_in_layers, num_out_layers, 1, 1, normalize=normalize)
        self.conv2 = Conv(num_out_layers, num_out_layers, 3, stride, normalize=normalize)

        total_num_out_layers = 4 * num_out_layers
        self.conv3 = nn.Conv2d(num_out_layers, total_num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, total_num_out_layers, kernel_size=1, stride=stride)

        if normalize:
            self.normalize = normalize(total_num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        # do_proj = True

        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x

        if self.normalize:
            shortcut = self.normalize(x_out + shortcut)
        else:
            shortcut = x_out + shortcut

        return F.elu(shortcut, inplace=True)


class ResConvBasic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride, normalize=None):
        super(ResConvBasic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = Conv(num_in_layers, num_out_layers, 3, stride, normalize=normalize)
        self.conv2 = Conv(num_out_layers, num_out_layers, 3, 1, normalize=normalize)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)

        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

    def forward(self, x):
        do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        # do_proj = True

        x_out = self.conv1(x)
        x_out = self.conv2(x_out)

        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x

        if self.normalize:
            shortcut = self.normalize(x_out + shortcut)
        else:
            shortcut = x_out + shortcut

        return F.elu(shortcut, inplace=True)


def ResBlock(num_in_layers, num_out_layers, num_blocks, stride, normalize=None):
    layers = []
    layers.append(ResConv(num_in_layers, num_out_layers, stride, normalize=normalize))
    for i in range(1, num_blocks - 1):
        layers.append(ResConv(4 * num_out_layers, num_out_layers, 1, normalize=normalize))
    layers.append(ResConv(4 * num_out_layers, num_out_layers, 1, normalize=normalize))
    return nn.Sequential(*layers)


def ResBlockBasic(num_in_layers, num_out_layers, num_blocks, stride, normalize=None):
    layers = []
    layers.append(ResConvBasic(num_in_layers, num_out_layers, stride, normalize=normalize))
    for i in range(1, num_blocks):
        layers.append(ResConvBasic(num_out_layers, num_out_layers, 1, normalize=normalize))
    return nn.Sequential(*layers)


class Upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale, normalize=None):
        super(Upconv, self).__init__()
        self.scale = scale
        self.conv1 = Conv(num_in_layers, num_out_layers, kernel_size, 1, normalize=normalize)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class GetDisp(nn.Module):
    def __init__(self, num_in_layers, num_out_layers=2, normalize=None):
        super(GetDisp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=3, stride=1)

        if normalize:
            self.normalize = normalize(num_out_layers)
        else:
            self.normalize = normalize

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        if self.normalize:
            x = self.normalize(x)
        return 0.3 * self.sigmoid(x)
