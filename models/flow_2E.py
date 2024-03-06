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
        self.conv1 = nn.Sequential(nn.Conv2d(9, 32, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))

        self.imconv4 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.combine4 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.imconv8 = nn.Sequential(nn.Conv2d(9, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.combine8 = nn.Sequential(nn.Conv2d(128*2, 128, kernel_size=1, stride=1, padding=0),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True))

        self.imconv16 = nn.Sequential(nn.Conv2d(9, 256, kernel_size=3, stride=1, padding=1),
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

        self.flow_head = nn.Sequential(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 4, kernel_size=5, stride=1, padding=2))

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, ldrs, expos):
        prev, cur, nxt = ldrs
        p_exp, c_exp, n_exp = expos
        cur_adj_exp = adj_expo_ldr_to_ldr(cur, c_exp, p_exp)

        p4 = F.avg_pool2d(prev, 4)
        c4 = F.avg_pool2d(cur_adj_exp, 4)
        n4 = F.avg_pool2d(nxt, 4)

        p8 = F.avg_pool2d(prev, 8)
        c8 = F.avg_pool2d(cur_adj_exp, 8)
        n8 = F.avg_pool2d(nxt, 8)

        p16 = F.avg_pool2d(prev, 16)
        c16 = F.avg_pool2d(cur_adj_exp, 16)
        n16 = F.avg_pool2d(nxt, 16)

        x = torch.cat((prev, cur_adj_exp, nxt), dim=1)

        x2 = self.conv1(x)
        x2 = self.layer1(x2)
        x4 = self.layer2(x2)
        x4_ = self.imconv4(torch.cat((p4, c4, n4), dim=1))
        x4 = self.combine4(torch.cat((x4, x4_), dim=1))

        x8 = self.layer3(x4)
        x8_ = self.imconv8(torch.cat((p8, c8, n8), dim=1))
        x8 = self.combine8(torch.cat((x8, x8_), dim=1))

        x16 = self.layer4(x8)
        x16_ = self.imconv16(torch.cat((p16, c16, n16), dim=1))
        x16 = self.combine16(torch.cat((x16, x16_), dim=1))

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

        p_flow4 = flow4[:,0:2].clamp(-100, 100)
        n_flow4 = flow4[:,2:].clamp(-100, 100)

        p_flow = F.interpolate(p_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0
        n_flow = F.interpolate(n_flow4, scale_factor=4, mode='bilinear', align_corners=True) * 4.0

        return [p_flow, n_flow]