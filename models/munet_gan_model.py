# skeleton taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import torch.nn as nn
import torch



class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=4, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(out_ch)
            )


    def forward(self, x):
        return self.block(x)


class DecBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=4, pad=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel, stride=stride, padding=pad),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self, depth=1, enc_chs=(3, 32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32, 1), kernel=5, stride=2, enc_pad=3, dec_pad=1):
        super().__init__()
        
        inenc = nn.Sequential(
            nn.Conv2d(enc_chs[0], enc_chs[1], kernel_size=kernel, stride=2, padding=enc_pad),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5)
            )
            
        
        if depth == 1:
            outdec = nn.Sequential(
                nn.ConvTranspose2d(2 * dec_chs[-2], dec_chs[-1], kernel_size=kernel, stride=stride, padding=dec_pad),
                nn.Sigmoid()
                )
        else:
            outdec = nn.Sequential(
                nn.ConvTranspose2d(dec_chs[-1], dec_chs[-1], kernel_size=kernel, stride=1, padding=1),
                nn.Sigmoid()
                )
        
        self.encs = []
        self.decs = []
        
        for k, (ch1, ch2) in enumerate(zip(enc_chs[: -1], enc_chs[1:])):
            blocks = []
            
            if k == 0:
                blocks.append(inenc)
            else:
                blocks.append(EncBlock(ch1, ch2, kernel=kernel, stride=stride, pad=enc_pad))
                
            
            for i in range(depth - 1):
                blocks.append(EncBlock(ch2, ch2, kernel=kernel, stride=1, pad=1))
                
            self.encs.append(nn.ModuleList(blocks))
            
        dec_len = len(dec_chs)
            
        for k, (ch1, ch2) in  enumerate(zip(dec_chs[:-1], dec_chs[1:])):
            blocks = []
            
            if k == 0:
                blocks.append(DecBlock(ch1, ch2, kernel=kernel, stride=stride, pad=dec_pad))
            elif k == dec_len - 2 and depth == 1:
                blocks.append(outdec)
            else:
                blocks.append(DecBlock(2 * ch1, ch2, kernel=kernel, stride=stride, pad=dec_pad))
    
                
            
            for i in range(depth - 1):
                if k == dec_len - 2 and i == depth - 2:
                    blocks.append(outdec)
                else:
                    blocks.append(DecBlock(ch2, ch2, kernel=kernel, stride=1, pad=1))
            
            self.decs.append(nn.ModuleList(blocks))
        
        self.encs = nn.ModuleList(self.encs)
        self.decs = nn.ModuleList(self.decs)
            
                

    def forward(self, x):
        in_size = x.size()[2]
        
        store = []
        
        for blocks in self.encs:
            for block in blocks:
                x = block(x)
            
            store.append(x)
            
        store = store[:-1][::-1]
        
        for i, blocks in enumerate(self.decs):
            if i > 0:
                store_size = store[i-1].size(2)
                x = x[:, :, 0:store_size, 0:store_size]
                x = torch.cat([x, store[i-1]], dim=1)
            
            for block in blocks:
                x = block(x)

        x = x[:, :, 0:in_size, 0:in_size]
        return x


class Discriminator(nn.Module):
    def __init__(self, depth=1, enc_chs=(4, 32, 64, 128, 256, 512), kernel=5, stride=2, enc_pad=3, dec_pad=1, mode='linear', linear_layers=2, hidden_nodes = 512, in_size=400):
        super().__init__()
        
        assert mode in ['conv', 'linear'], 'not valid mode'
        self.mode = mode

        inenc = nn.Sequential(
            nn.Conv2d(enc_chs[0], enc_chs[1], kernel_size=kernel, stride=2, padding=enc_pad),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5)
            )
        
        if mode == 'conv':
            if depth == 1:
                outenc = nn.Sequential(
                    nn.Conv2d(enc_chs[-1], 1, kernel_size=3, stride=2, padding=1),
                    nn.Sigmoid()
                )
            else:
                outenc = nn.Sequential(
                    nn.ConvTranspose2d(1, 1, kernel_size=kernel, stride=1, padding=1),
                    nn.Sigmoid()
                    )
        elif mode == 'linear':
            if depth == 1:
                outenc = nn.Sequential(
                    nn.ConvTranspose2d(enc_chs[-1], 1, kernel_size=kernel, stride=1, padding=1),
                    nn.Dropout2d(0.5)
                    )
            else:
                outenc = nn.Sequential(
                    nn.ConvTranspose2d(1, 1, kernel_size=kernel, stride=1, padding=1),
                    nn.Dropout2d(0.5)
                    )
            
        
        self.encs = []
        
        enc_len = len(enc_chs)
        
        for k, (ch1, ch2) in enumerate(zip(enc_chs[: -1], enc_chs[1:])):
            blocks = []
            
            if k == 0:
                blocks.append(inenc)
            else:
                blocks.append(EncBlock(ch1, ch2, kernel=kernel, stride=stride, pad=enc_pad))
            
            for i in range(depth - 1):
                blocks.append(EncBlock(ch2, ch2, kernel=kernel, stride=1, pad=1))
                
            self.encs.append(nn.ModuleList(blocks))
        
        if mode == 'conv':
            blocks = []
            blocks.append(outenc)
            self.encs.append(nn.ModuleList(blocks))
        
        self.encs = nn.ModuleList(self.encs)
        
        x = torch.empty(1, 4, in_size, in_size)
        
        
        for blocks in self.encs:
            for block in blocks:
                
                x = block(x)
                
        in_nodes = x.size()[1] * x.size()[2] * x.size()[3]
        
        self.linear = []
        
        if linear_layers == 1:
            in_linear = nn.Sequential(
                nn.Linear(in_nodes, 1),
                nn.Sigmoid()
                )
        else:
            in_linear = nn.Sequential(
                nn.Linear(in_nodes, hidden_nodes),
                nn.LeakyReLU(),
                nn.Dropout(0.5)
                )
            
        self.linear.append(in_linear)
        
        for i in range(linear_layers - 2):
            self.linear.append(
                nn.Sequential(
                    nn.Linear(hidden_nodes, hidden_nodes),
                    nn.LeakyReLU(),
                    nn.Dropout(0.5)
                    )
                )
            
        if linear_layers > 1:
            self.linear.append(
                nn.Sequential(
                    nn.Linear(hidden_nodes, 1),
                    nn.Sigmoid()
                    )
                )
            
        self.linear = nn.ModuleList(self.linear)
        
        
        
        

    def forward(self, segmentation, satellite_image):
        x = torch.cat([segmentation, satellite_image], dim=1)

        for blocks in self.encs:
            for block in blocks:
                x = block(x)
                
        if self.mode == 'conv':
            x = x[:, :, 0:1, 0:1]
        elif self.mode == 'linear':
            x = torch.flatten(x, start_dim=1)
            for block in self.linear:
                x = block(x)

        return x
