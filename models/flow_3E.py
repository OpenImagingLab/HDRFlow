import torch
import torch.nn as nn
from .network_utils import *
import torch.nn.functional as F

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class Flow_Net(nn.Module):
    """ for 2-exposure """
    def __init__(self):
        super(Flow_Net, self).__init__()

        self.norm_fn = 'batch'
        self.in_planes = 32
        self.conv1 = nn.Sequential(nn.Conv2d(18, 32, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))

        self.imconv4 = nn.Sequential(nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.combine4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.imconv8 = nn.Sequential(nn.Conv2d(18, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.combine8 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.imconv16 = nn.Sequential(nn.Conv2d(18, 256, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True))

        self.combine16 = nn.Sequential(nn.Conv2d(256*2, 256, kernel_size=1, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(32,  stride=1) # 1/2
        self.layer2 = self._make_layer(64, stride=2) # 1/4
        self.layer3 = self._make_layer(128, stride=2) # 1/8
        self.layer4 = self._make_layer(256, stride=2) # 1/16

        self.upconv4 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.iconv3 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True))

        self.upconv3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.iconv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))

        self.conv7x7 = nn.Conv2d(256, 256, 7, stride=1, padding=7//2, groups=256)
        self.conv9x9 = nn.Conv2d(256, 256, 9, stride=1, padding=9//2, groups=256)
        self.conv11x11 = nn.Conv2d(256, 256, 11, stride=1, padding=11//2, groups=256)

        self.merge_features = nn.Conv2d(256*3, 256, 1, stride=1, padding=0)

        self.flow_head = nn.Sequential(nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 8, kernel_size=5, stride=1, padding=2))

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, ldrs, expos):
        
        p2, p1, c, n1, n2 = ldrs
        p2_exp, p1_exp, c_exp, n1_exp, n2_exp = expos
        c_adj_p2 = adj_expo_ldr_to_ldr(c, c_exp, p2_exp)
        c_adj_p1 = adj_expo_ldr_to_ldr(c, c_exp, p1_exp)

        p2_4 = F.avg_pool2d(p2, 4)
        c_adj_p2_4 = F.avg_pool2d(c_adj_p2, 4)
        n1_4 = F.avg_pool2d(n1, 4)

        p1_4 = F.avg_pool2d(p1, 4)
        c_adj_p1_4 = F.avg_pool2d(c_adj_p1, 4)
        n2_4 = F.avg_pool2d(n2, 4)

        p2_8 = F.avg_pool2d(p2, 8)
        c_adj_p2_8 = F.avg_pool2d(c_adj_p2, 8)
        n1_8 = F.avg_pool2d(n1, 8)

        p1_8 = F.avg_pool2d(p1, 8)
        c_adj_p1_8 = F.avg_pool2d(c_adj_p1, 8)
        n2_8 = F.avg_pool2d(n2, 8)

        p2_16 = F.avg_pool2d(p2, 16)
        c_adj_p2_16 = F.avg_pool2d(c_adj_p2, 16)
        n1_16 = F.avg_pool2d(n1, 16)

        p1_16 = F.avg_pool2d(p1, 16)
        c_adj_p1_16 = F.avg_pool2d(c_adj_p1, 16)
        n2_16 = F.avg_pool2d(n2, 16)

        x = torch.cat((p2, c_adj_p2, n1, p1, c_adj_p1, n2), dim=1)

        x2 = self.conv1(x)
        x2 = self.layer1(x2)
        x4 = self.layer2(x2)
        x4_ = self.imconv4(torch.cat((p2_4, c_adj_p2_4, n1_4, p1_4, c_adj_p1_4, n2_4), dim=1))
        x4 = self.combine4(torch.cat((x4, x4_), dim=1))

        x8 = self.layer3(x4)
        x8_ = self.imconv8(torch.cat((p2_8, c_adj_p2_8, n1_8, p1_8, c_adj_p1_8, n2_8), dim=1))
        x8 = self.combine8(torch.cat((x8, x8_), dim=1))

        x16 = self.layer4(x8)
        x16_ = self.imconv16(torch.cat((p2_16, c_adj_p2_16, n1_16, p1_16, c_adj_p1_16, n2_16), dim=1))
        x16 = self.combine16(torch.cat((x16, x16_), dim=1))
        # x16 = self.conv2(x16)

        x16_7x7 = self.conv7x7(x16)
        x16_9x9 = self.conv9x9(x16)
        x16_11x11 = self.conv11x11(x16)
        lk_features = torch.cat((x16_7x7, x16_9x9, x16_11x11), dim=1)
        lk_features = self.merge_features(lk_features)
        x16 = F.relu(x16+lk_features)

        x16_up = self.upconv4(x16)
        x8 = self.iconv3(torch.cat((x8, x16_up), dim=1))

        x8_up = self.upconv3(x8)
        x4 = self.iconv2(torch.cat((x4, x8_up), dim=1))
        flow4 = self.flow_head(x4)

        p2_flow4 = flow4[:,:2].clamp(-64, 64)
        n1_flow4 = flow4[:,2:4].clamp(-64, 64)

        p1_flow4 = flow4[:,4:6].clamp(-64, 64)
        n2_flow4 = flow4[:,6:].clamp(-64, 64)

        p2_flow = F.interpolate(p2_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        n1_flow = F.interpolate(n1_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0

        p1_flow = F.interpolate(p1_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        n2_flow = F.interpolate(n2_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0

        return [p2_flow, n1_flow, p1_flow, n2_flow]