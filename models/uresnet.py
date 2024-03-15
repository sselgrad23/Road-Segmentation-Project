
import torch.nn as nn
import torch

class In(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=pad),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)

class Out(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=1, padding=pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)

class DoubleSkip(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1):
        if pad is None:
            pad = kernel
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
        )

        self.block2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        x0 = x

        x = self.block1(x)

        x += x0

        x = self.block2(x)

        return x

class Down(nn.Module):
    def __init__(self, in_ch, out_ch,  kernel=3, stride=2, pad=1):
        if pad is None:
            pad = kernel
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pad=0, stride=2, skip=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.block(x)

class UResNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = 64
        rep = 1
        depth = 4

        k = base

        encs = []
        downs = []
        decs = []
        ups = []

        self.inb = In(3, base)

        for i in range(depth):
            blocks = []
            for j in range(rep):
                blocks.append(DoubleSkip(k, k))
            downs.append(Down(k, k*2))


            encs.append(nn.ModuleList(blocks))

            k = k * 2

        self.mid = DoubleSkip(k, k)

        for i in range(depth):
            ups.append(Up(k, int(k / 2)))
            k = int(k / 2)
            blocks = []

            for j in range(rep):
                blocks.append(DoubleSkip(2 * k, k))



            decs.append(nn.ModuleList(blocks))



        self.encs = nn.ModuleList(encs)
        self.decs = nn.ModuleList(decs)

        self.ups = nn.ModuleList(ups)
        self.downs = nn.ModuleList(downs)

        self.outb = Out(base, 1)

    def forward(self, x):
        in_size = x.size()[2]

        x = self.inb(x)

        stores = []

        for blocks, down in zip(self.encs, self.downs):
            for block in blocks:
                x = block(x)

            stores.append(x)
            x = down(x)

        stores = stores[::-1]
        x = self.mid(x)

        for blocks, up, store in zip(self.decs, self.ups, stores):
            x = up(x)
            store_size = store.size(2)
            print(store.size())
            print(x.size())
            x = x[:, :, 0:store_size, 0:store_size]
            x = torch.cat([x, store], dim=1)

            for block in blocks:
                x = block(x)



        x = self.outb(x)

        x = x[:, :, 0:in_size, 0:in_size]
        return x
