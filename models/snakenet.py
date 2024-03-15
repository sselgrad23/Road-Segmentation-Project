# skeleton taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch.nn as nn
import torch


class Double(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, firstkernel=None, pad=1):
        if firstkernel is None:
            firstkernel = kernel
        else:
            pad = int(firstkernel / 2)
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, bias=False, kernel_size=firstkernel, stride=1, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, bias=False, kernel_size=kernel, stride=1, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, pad=None):
        if pad is None:
            pad = kernel
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, bias=False, kernel_size=kernel, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, ch, kernel=3, stride=2, pad=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch, out_channels=ch, bias=False, kernel_size=kernel, stride=stride, padding=pad),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)


class SnakeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.t1 = Double(3, 64, firstkernel=7)
        self.d1 = Down(64, 128)
        self.t2 = Double(128, 128)
        self.d2 = Down(128, 256)
        self.t3 = Double(256, 256)
        self.u3 = Up(256)

        self.t4 = Double(256 + 128, 128)
        self.d4 = Down(128, 256)
        self.t5 = Double(256 + 256, 256)
        self.d5 = Down(256, 512)
        self.t6 = Double(512, 512)
        self.u6 = Up(512)

        self.t7 = Double(512 + 256, 256)
        self.d7 = Down(256, 512)
        self.t8 = Double(512 + 512, 512)
        self.d8 = Down(512, 1024)

        self.t9 = Double(1024, 1024)
        self.u9 = Up(1024)

        self.t10 = Double(512 + 1024, 512)
        self.u10 = Up(512)
        self.t11 = Double(256 + 512, 256)
        self.d11 = Down(256, 512)

        self.t12 = Double(512 + 512, 512)
        self.u12 = Up(512)
        self.t13 = Double(256 + 512, 256)
        self.u13 = Up(256)
        self.t14 = Double(128 + 256, 128)
        self.d14 = Down(128, 256)

        self.t15 = Double(256 + 256, 256)
        self.u15 = Up(256)
        self.t16 = Double(128 + 256, 128)
        self.u16 = Up(128)
        self.t17 = Double(128 + 64, 64)

        self.out = self.block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x0):
        x0s = x0.size()[2]

        x1 = self.t1(x0)
        x = self.d1(x1)
        x2 = self.t2(x)
        x = self.d2(x2)
        x3 = self.t3(x)

        x = self.u3(x3)
        x2s = x2.size()[2]
        x = torch.cat([x2, x[:, :, 0:x2s, 0:x2s]], dim=1)
        x4 = self.t4(x)
        x = self.d4(x4)

        x3s = x3.size()[2]
        x = torch.cat([x3, x[:, :, 0:x3s, 0:x3s]], dim=1)
        x5 = self.t5(x)
        x = self.d5(x5)

        x6 = self.t6(x)
        x = self.u6(x6)

        x5s = x5.size()[2]
        x = torch.cat([x5, x[:, :, 0:x5s, 0:x5s]], dim=1)
        x7 = self.t7(x)
        x = self.d7(x7)

        x6s = x6.size()[2]
        x = torch.cat([x6, x[:, :, 0:x6s, 0:x6s]], dim=1)
        x8 = self.t8(x)
        x = self.d8(x8)

        x9 = self.t9(x)
        x = self.u9(x9)

        x8s = x8.size()[2]
        x = torch.cat([x8, x[:, :, 0:x8s, 0:x8s]], dim=1)
        x10 = self.t10(x)
        x = self.u10(x10)

        x7s = x7.size()[2]
        x = torch.cat([x7, x[:, :, 0:x7s, 0:x7s]], dim=1)
        x11 = self.t11(x)
        x = self.d11(x11)

        x10s = x10.size()[2]
        x = torch.cat([x10, x[:, :, 0:x10s, 0:x10s]], dim=1)
        x12 = self.t12(x)
        x = self.u12(x12)

        x11s = x11.size()[2]
        x = torch.cat([x11, x[:, :, 0:x11s, 0:x11s]], dim=1)
        x13 = self.t13(x)
        x = self.u13(x13)

        x4s = x4.size()[2]
        x = torch.cat([x4, x[:, :, 0:x4s, 0:x4s]], dim=1)
        x14 = self.t14(x)
        x = self.d14(x14)

        x13s = x13.size()[2]
        x = torch.cat([x13, x[:, :, 0:x13s, 0:x13s]], dim=1)
        x15 = self.t15(x)
        x = self.u15(x15)

        x14s = x14.size()[2]
        x = torch.cat([x14, x[:, :, 0:x14s, 0:x14s]], dim=1)
        x16 = self.t16(x)
        x = self.u16(x16)

        x1s = x1.size()[2]
        x = torch.cat([x1, x[:, :, 0:x1s, 0:x1s]], dim=1)
        x17 = self.t17(x)
        x = self.out(x17)

        x = x[:, :, 0:x0s, 0:x0s]
        return x
