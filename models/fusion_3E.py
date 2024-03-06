import torch
import torch.nn as nn
from .network_utils import *
class Downsample(nn.Module):
    def __init__(self, c_in, c_out=512, use_bn=True, afunc='LReLU'):
        super(Downsample, self).__init__()

        self.layers = torch.nn.Sequential(
            conv_layer(c_in,     c_out//4, k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn), 
            conv_layer(c_out//4, c_out//4, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            conv_layer(c_out//4, c_out//2, k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn),
            conv_layer(c_out//2, c_out//2, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            conv_layer(c_out//2, c_out,    k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn), 
            conv_layer(c_out,    c_out,    k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
        )

    def forward(self, inputs):
        out = self.layers(inputs)
        return out

class Upsample(nn.Module):
    def __init__(self, c_in, c_out, use_bn=True, afunc='LReLU'):
        super(Upsample, self).__init__()
        last_c = max(128, c_in // 8)
        self.layers = torch.nn.Sequential(
            deconv_layer(c_in, c_in//2, use_bn=use_bn, afunc=afunc),
            conv_layer(c_in//2, c_in//2, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            deconv_layer(c_in//2,  c_in//2, use_bn=use_bn, afunc=afunc),
            conv_layer(c_in//2, c_in//2, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn),
            deconv_layer(c_in//2, last_c, use_bn=use_bn, afunc=afunc),
            output_conv(last_c, c_out, k=3, stride=1, pad=1),
        )

    def forward(self, inputs):
        out = self.layers(inputs)
        out = torch.sigmoid(out)
        return out

class Fusion_Net(nn.Module):
    def __init__(self, c_in=30, c_out=15, c_mid=256):
        super(Fusion_Net, self).__init__()

        self.conv1 = torch.nn.Sequential(
            nn.Conv2d(c_in, c_mid//4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid//4),
            nn.LeakyReLU(0.1, inplace=True))       
        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(c_mid//4, c_mid//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid//4),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(c_mid//4, c_mid//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid//2),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv4 = torch.nn.Sequential(
            nn.Conv2d(c_mid//2, c_mid//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid//2),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv5 = torch.nn.Sequential(
            nn.Conv2d(c_mid//2, c_mid, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv6 = torch.nn.Sequential(
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv7 = torch.nn.Sequential(
            nn.ConvTranspose2d(c_mid, c_mid//2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_mid//2),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv8 = torch.nn.Sequential(
            nn.Conv2d(c_mid, c_mid//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid//2),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv9 = torch.nn.Sequential(
            nn.ConvTranspose2d(c_mid//2, c_mid//4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c_mid//4),
            nn.LeakyReLU(0.1, inplace=True))
        self.conv10 = torch.nn.Sequential(
            nn.Conv2d(c_mid//2, c_mid//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c_mid//4),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv11 = torch.nn.Sequential(
            nn.ConvTranspose2d(c_mid//4, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True))

        self.conv12 = torch.nn.Sequential(
            nn.Conv2d(64, c_out, kernel_size=3, stride=1, padding=1))


        self.merge_HDR = MergeHDRModule()

    def forward(self, inputs, hdrs):
        pred = OrderedDict()
        n, c, h, w = inputs.shape
        ### 1/2
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)

        ### 1/4
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        ### 1/8
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        ### 1/4
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(torch.cat((conv7, conv4), dim=1))
        ### 1/2
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(torch.cat((conv9, conv2), dim=1))
        ### output
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)
        weights = torch.sigmoid(conv12)

        ws = torch.split(weights, 1, 1)       
        hdr, weights = self.merge_HDR(ws, hdrs)
        return hdr