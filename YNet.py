import torch
import torch.nn as nn
import torch.nn.functional as F


'''
After network construction, the Y-Net’s parameters are trained based on a training dataset.
To accelerate convergence of Y-Net, C1 1 -C16 1 ’s parameters are pre-trained. An auxiliary
network, containing C1 1 -C16 1 , B1 1 -B5 1 , P1 1 -P5 1 , and three fully connected layers, is con-
structed. The auxiliary network accepts an input image patch with size of 9 × 9 pixels, and
outputs a label for this patch
from the paper...
?????????????????
'''

class YNet(nn.Module):
    def __init__(self, input_dim, img_channels, version=1):
        super().__init__()
        self.version = version

        self.y1 = Y1(input_dim=input_dim, img_channels=img_channels, version=self.version)

        self.y2 = Y2()

        self.y3 = Y3()

    def forward(self, x):
        x1 = self.y1(x)
        x2 = self.y2(x)

        # x = torch.cat([x1, x2], dim=1)
        x = self.y3(x1, x2)

        return x


# Y1 is for the most part a U-Net architecture --- the last layers seem to be a simplified upsampling...
class Y1(nn.Module):
    # Assuming square inputs
    def __init__(self, input_dim, img_channels, version=1):
        super().__init__()
        self.input_dim = input_dim
        self.version = version
        # TODO check if padding the input to the next largest values divisible by 32 is ok?
        #  (makes everything regarding padding easy)
        #  this assumes that input_dim is a multiple of 2!

        if input_dim % 32 == 0:
            self.pad = 1
        else:
            self.padded_size = ((input_dim // 32) + 1) * 32
            self.pad = (self.padded_size - input_dim) // 2 + 1

        self.double_conv1 = double_conv(c_in=img_channels, c_mid=64, c_out=64, kernel_size=3,
                                        padding=self.pad, padding2=1)  # C1 B1 C2
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # P1

        self.double_conv2 = double_conv(c_in=64, c_mid=128, c_out=128, kernel_size=3, padding=1)  # C3 B2 C4
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # P2

        self.double_conv3 = double_conv(c_in=128, c_mid=256, c_out=256, kernel_size=3, padding=1,
                                        repeat_first=True)  # C5 C6 B3 C7
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # P3

        self.double_conv4 = double_conv(c_in=256, c_mid=512, c_out=512, kernel_size=3, padding=1,
                                        repeat_first=True)  # C8 C9 B4 C10
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # P4

        self.double_conv5 = double_conv(c_in=512, c_mid=512, c_out=512, kernel_size=3, padding=1,
                                        repeat_first=True)  # C11 C12 B5 C13
        self.mpool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # P5

        # C14 C15 C16
        self.middle_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=4096, out_channels=2, kernel_size=1, padding=0),
            nn.ReLU(),
        )

        # TODO: check if the ConvTranspose2d is correct or of we should use the Upsample() layer
        # deconv1 = nn.Upsample()      # D1
        self.deconv1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)  # D1
        # TODO: now we put the output of C13 into conv_after_deconv1 to reduce the number of channels to 2
        self.conv_before_concat1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0),
            nn.ReLU()
        )  # C17

        if self.version in [2, 3]:
            self.conv_after_deconv_concat1 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU()
            )

        # crop1 =         # R1
        # sum1 =          # S1

        # TODO: here we concatenate the skip with the normal

        self.deconv2 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)  # D2

        self.conv_before_concat2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2, kernel_size=1, padding=0),  # C17
            nn.ReLU()
        )  # C18

        if self.version in [2, 3]:
            self.conv_after_deconv_concat2 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU()
            )
        # crop2 =         # R2
        # sum2 =          # S2

        # TODO: here we concatenate the skip with the normal

        self.deconv3 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)  # D3
        if self.version == 2:
            self.conv_after_deconv3 = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
                nn.ReLU()
            )

        if self.version == 3:
            self.conv_before_concat3 = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, padding=0),  # C17
                nn.ReLU()
            )
            self.conv_after_deconv_concat3 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)  # D4
        if self.version == 2:
            self.conv_after_deconv4 = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
                nn.ReLU()
            )
        if self.version == 3:
            self.conv_before_concat4 = nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1, padding=0),  # C17
                nn.ReLU()
            )
            self.conv_after_deconv_concat4 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.deconv4 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)  # D4

        self.deconv5 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)  # D5
        if self.version == 2:
            self.conv_after_deconv5 = nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1),
                nn.ReLU()
            )
        if self.version == 3:
            self.conv_before_concat5 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, padding=0),  # C17
                nn.ReLU()
            )
            self.conv_after_deconv_concat5 = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.deconv5 = nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=4, stride=2, padding=1)  # D5

        # TODO: put this crop in the forward!!!
        # neg_pad = (self.padded_size - self.input_dim) // 2
        # crop3 = F.pad(x, (-neg_pad, -neg_pad, -neg_pad, -neg_pad))        # R2

        ### TODO: Skip connections: ....
        ### TODO: why do they only do deconvolutions at some point without convolutions and skip connections???

    def forward(self, x):
        # print(f"shape of x1: {x.shape}")
        x = self.double_conv1(x)
        # print(f"shape of x2: {x.shape}")
        if self.version == 3:
            skip1 = x
        x = self.mpool1(x)
        # print(f"shape of x3: {x.shape}")

        x = self.double_conv2(x)
        # print(f"shape of x4: {x.shape}")
        if self.version == 3:
            skip2 = x
        x = self.mpool2(x)
        # print(f"shape of x5: {x.shape}")

        x = self.double_conv3(x)
        # print(f"shape of x6: {x.shape}")
        if self.version == 3:
            skip3 = x
        x = self.mpool3(x)
        # print(f"shape of x7: {x.shape}")

        x = self.double_conv4(x)
        # print(f"shape of x8: {x.shape}")
        skip4 = x
        x = self.mpool4(x)
        # print(f"shape of x9: {x.shape}")

        x = self.double_conv5(x)
        # print(f"shape of x10: {x.shape}")
        skip5 = x
        x = self.mpool5(x)
        # print(f"shape of x11: {x.shape}")

        x = self.middle_conv(x)
        # print(f"shape of x12: {x.shape}")

        x = self.deconv1(x)
        # print(f"shape of x13: {x.shape}")
        skip5 = self.conv_before_concat1(skip5)
        # print(f"shape of x14: {x.shape}")
        x = torch.cat([x, skip5], dim=1)
        # print(f"shape of x15: {x.shape}")

        if self.version in [2, 3]:
            x = self.conv_after_deconv_concat1(x)
            # print(f"shape of new---15: {x.shape}")

        x = self.deconv2(x)
        # print(f"shape of x16: {x.shape}")
        skip4 = self.conv_before_concat2(skip4)
        x = torch.cat([x, skip4], dim=1)
        # print(f"shape of x17: {x.shape}")
        if self.version in [2, 3]:
            x = self.conv_after_deconv_concat2(x)
            # print(f"shape of new---17: {x.shape}")

        x = self.deconv3(x)
        # print(f"shape of x18: {x.shape}")
        if self.version == 2:
            x = self.conv_after_deconv3(x)
        elif self.version == 3:
            skip3 = self.conv_before_concat3(skip3)
            x = torch.cat([x, skip3], dim=1)

            x = self.conv_after_deconv_concat3(x)

            # print(f"shape of new---18: {x.shape}")

        x = self.deconv4(x)
        # print(f"shape of x19: {x.shape}")
        if self.version == 2:
            x = self.conv_after_deconv4(x)
            # print(f"shape of new---19: {x.shape}")
        elif self.version == 3:
            skip2 = self.conv_before_concat4(skip2)
            x = torch.cat([x, skip2], dim=1)
            x = self.conv_after_deconv_concat4(x)

        x = self.deconv5(x)
        # print(f"shape of x20: {x.shape}")

        if self.version == 2:
            x = self.conv_after_deconv5(x)
            # print(f"shape of new---20: {x.shape}")
        elif self.version == 3:
            skip1 = self.conv_before_concat5(skip1)
            x = torch.cat([x, skip1], dim=1)
            x = self.conv_after_deconv_concat5(x)

        neg_pad = (self.padded_size - self.input_dim) // 2
        x = F.pad(x, (-neg_pad, -neg_pad, -neg_pad, -neg_pad))
        # print(f"shape of x21: {x.shape}")

        return x


# Y2 is the large convolutional Module
class Y2(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: C1-C6 & C13 are ks3 + relu and C7-C12 are ks5 + relu
        self.conv1 = conv2d_relu(c_in=3, c_out=16, kernel_size=3, padding=1)  # C1
        self.conv2 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C2
        self.conv3 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C3
        self.conv4 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C4
        self.conv5 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C5
        self.conv6 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C6
        self.conv7 = conv2d_relu(c_in=16, c_out=32, kernel_size=5, padding=2)  # C7
        self.conv8 = conv2d_relu(c_in=32, c_out=32, kernel_size=5, padding=2)  # C8
        self.conv9 = conv2d_relu(c_in=32, c_out=32, kernel_size=5, padding=2)  # C9
        self.conv10 = conv2d_relu(c_in=32, c_out=64, kernel_size=5, padding=2)  # C10
        self.conv11 = conv2d_relu(c_in=64, c_out=64, kernel_size=5, padding=2)  # C11
        self.conv12 = conv2d_relu(c_in=64, c_out=64, kernel_size=5, padding=2)  # C12
        # TODO: check if this last step makes sense for our "improved implementation"
        self.conv13 = conv2d_relu(c_in=64, c_out=2, kernel_size=1, padding=0)  # C13

    def forward(self, x):
        # print(f"shape of x22: {x.shape}")
        x = self.conv1(x)
        # print(f"shape of x23: {x.shape}")
        x = self.conv2(x)
        # print(f"shape of x24: {x.shape}")
        x = self.conv3(x)
        # print(f"shape of x25: {x.shape}")
        x = self.conv4(x)
        # print(f"shape of x26: {x.shape}")
        x = self.conv5(x)
        # print(f"shape of x27: {x.shape}")
        x = self.conv6(x)
        # print(f"shape of x28: {x.shape}")
        x = self.conv7(x)
        # print(f"shape of x29: {x.shape}")
        x = self.conv8(x)
        # print(f"shape of x30: {x.shape}")
        x = self.conv9(x)
        # print(f"shape of x31: {x.shape}")
        x = self.conv10(x)
        # print(f"shape of x32: {x.shape}")
        x = self.conv11(x)
        # print(f"shape of x33: {x.shape}")
        x = self.conv12(x)
        # print(f"shape of x34: {x.shape}")
        x = self.conv13(x)
        # print(f"shape of x35: {x.shape}")
        return x


# Y3 is the last part where the outputs of Y1 and Y2 are concatenated and the prediction is produced
class Y3(nn.Module):
    def __init__(self):
        super().__init__()


        self.conv1 = conv2d_relu(c_in=4, c_out=16, kernel_size=3, padding=1)  # C1
        self.conv2 = conv2d_relu(c_in=16, c_out=16, kernel_size=3, padding=1)  # C2
        self.conv3 = conv2d_relu(c_in=16, c_out=32, kernel_size=3, padding=1)  # C3
        self.conv4 = conv2d_relu(c_in=32, c_out=32, kernel_size=3, padding=1)  # C4
        # TODO: CHANGE BACK?
        # self.conv5 = conv2d_relu(c_in=32, c_out=2, kernel_size=1, padding=0)  # C5

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0) # C5
        # self.conv5 = conv2d_relu(c_in=32, c_out=1, kernel_size=1, padding=0)

        # TODO: Why the FUCK is the number of out channels of conv5 2???????????????????

    def forward(self, x1, x2):
        # print(f"shape of x36: {x1.shape}")
        # print(f"shape of x37: {x2.shape}")
        x = torch.cat([x1, x2], dim=1)
        # print(f"shape of x38: {x.shape}")
        x = self.conv1(x)
        # print(f"shape of x39: {x.shape}")
        x = self.conv2(x)
        # print(f"shape of x40: {x.shape}")
        x = self.conv3(x)
        # print(f"shape of x41: {x.shape}")
        x = self.conv4(x)
        # print(f"shape of x42: {x.shape}")
        x = self.conv5(x)
        # print(f"shape of x43: {x.shape}")
        return x


def conv2d_relu(c_in, c_out, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
    )


def double_conv(c_in, c_mid, c_out, kernel_size, padding, padding2=None, repeat_first=False):
    if repeat_first:
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_mid, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_mid, out_channels=c_mid, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(c_mid),
            nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=kernel_size, padding=padding) if padding2 is None else nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=kernel_size, padding=padding2),
            nn.ReLU(),
        )

    else:
        return nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_mid, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(c_mid),
            nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=kernel_size, padding=padding) if padding2 is None else nn.Conv2d(in_channels=c_mid, out_channels=c_out, kernel_size=kernel_size, padding=padding2),
            nn.ReLU(),
        )

