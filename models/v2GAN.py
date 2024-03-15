
# part of implementation from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch.nn as nn
import torch


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, downsampling=False, dropout=False):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Dropout() if dropout else nn.Identity(),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, stride= 2 if downsampling else 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout() if dropout else nn.Identity(),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        #print('dconv: ', x.shape)
        a = self.double_conv(x)
        #print('dconv: ', a.shape)
        return a


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, downsampling=True, dropout=dropout)
        )

    def forward(self, x):
        #print('down: ', x.shape)
        a = self.maxpool_conv(x)
        #print('down: ', a.shape)
        return a



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dropout=False):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ReduceChannelsConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReduceChannelsConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):

    def __init__(self, bilinear=False):
        super(Generator, self).__init__()
        self.n_channels = 3
        self.bilinear = bilinear
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128, dropout=True)
        self.down2 = Down(128, 256, dropout=True)
        self.down3 = Down(256, 512, dropout=True)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, dropout=True)
        self.up1 = Up(1024, 512 // factor, bilinear, dropout=True)
        self.up2 = Up(512, 256 // factor, bilinear, dropout=True)
        self.up3 = Up(256, 128 // factor, bilinear, dropout=True)
        self.up4 = Up(128, 64, bilinear, dropout=True)
        self.outc = OutConv(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        tanh_logits = self.tanh(logits)
        return (tanh_logits + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_channels = 4
        # first we massively downsample
        self.model = nn.Sequential(
            DoubleConv(self.n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 512),
            Down(512, 512),
            ReduceChannelsConv(512, 512),
            ReduceChannelsConv(512, 256),
            ReduceChannelsConv(256, 128),
            ReduceChannelsConv(128, 64),
            ReduceChannelsConv(64, 32),
            ReduceChannelsConv(32, 16),
            ReduceChannelsConv(16, 16),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        )

    def forward(self, segmentation, satellite_image):
        return self.model(torch.cat((segmentation, satellite_image), axis=1))
