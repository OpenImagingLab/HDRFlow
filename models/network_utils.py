"""
Util functions for network construction
"""
import os
import importlib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def import_network(net_name, arch_dir='models.archs', backup_module='networks'):
    print('%s/%s' % (arch_dir.replace('.', '/'), net_name))
    if os.path.exists('%s/%s.py' % (arch_dir.replace('.', '/'), net_name)):
        network_file = importlib.import_module('%s.%s' % (arch_dir, net_name))
    else:
            network_file = importlib.import_module('%s.%s' % (arch_dir, backup_module))
    network = getattr(network_file, net_name)
    return network

## Common Network Blocks
def activation(afunc='LReLU', inplace=True):
    if afunc == 'LReLU':
        return nn.LeakyReLU(0.1, inplace=inplace)
    elif afunc == 'LReLU02':
        return nn.LeakyReLU(0.2, inplace=inplace)
    elif afunc == 'ReLU':
        return nn.ReLU(inplace=inplace)
    elif afunc == 'Sigmoid':
        return nn.Sigmoid()
    elif afunc == 'Tanh':
        return nn.Tanh()
    else:
        raise Exception('Unknown activation function')

def conv_layer(cin, cout, k=3, stride=1, pad=-1, dilation=1, afunc='LReLU', use_bn=False, bias=True, inplace=True):
    if type(pad) != tuple:
        pad = pad if pad >= 0 else (k - 1) // 2
    block = [nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=bias, dilation=dilation)]
    if use_bn: 
        # print('=> convolutional layer with bachnorm')
        block += [nn.BatchNorm2d(cout)]
    if afunc != '':
        block += [activation(afunc, inplace=inplace)]
    return nn.Sequential(*block)

def deconv_layer(cin, cout, afunc='LReLU', use_bn=False, bias=False):
    block = [nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=bias)]
    if use_bn: 
        # print('=> deconvolutional layer with bachnorm')
        block += [nn.BatchNorm2d(cout)]
    block += [activation(afunc)]
    return nn.Sequential(*block)

def upconv_layer(cin, cout, afunc='LReLU', mode='bilinear', use_bn=False, bias=True):
    if mode == 'bilinear':
        block = [nn.Upsample(scale_factor=2, mode=mode, align_corners=True)]
    elif mode == 'nearest':
        block = [nn.Upsample(scale_factor=2, mode=mode)]
    else:
        raise Exception('Unknonw mode: %s' % mode)
    block += [nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=bias)]
    if use_bn: 
        # print('=> deconvolutional layer with bachnorm')
        block += [nn.BatchNorm2d(cout)]
    block += [activation(afunc)]
    return nn.Sequential(*block)

def up_nearest_layer(cin, cout, afunc='LReLU', use_bn=False, bias=True):
    block = [nn.Upsample(scale_factor=2, mode='nearest'), 
             nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=bias)]
    if use_bn: 
        # print('=> deconvolutional layer with bachnorm')
        block += [nn.BatchNorm2d(cout)]
    block += [activation(afunc)]
    return nn.Sequential(*block)

def output_conv(cin, cout, k=1, stride=1, pad=0, bias=True):
    pad = (k - 1) // 2
    return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=bias)
            )

def fc_layer(cin, cout):
    return nn.Sequential(
            nn.Linear(cin, cout),
            nn.LeakyReLU(0.1, inplace=True))

def down_block(cin, cout, k=3, layers=2, down='conv', afunc='LReLU', use_bn=True):
    block = []
    if down == 'conv':
        block += [conv_layer(cin, cout, k=3, stride=2, pad=1, afunc=afunc, use_bn=use_bn)] 
    elif down == 'max':
        block += [conv_layer(cin, cout, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn)]
        block += [torch.nn.MaxPool2d(2)]
    elif down == 'avg':
        block += [conv_layer(cin, cout, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn)]
        block += [torch.nn.AvgPool2d(2)]
    else:
        raise Exception('Unknown downsample mode %s' % (down))
    for i in range(1, layers):
        block += [conv_layer(cout, cout, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn)]
    return nn.Sequential(*block)

def up_block(cin, cout, layers=2, up='bilinear', afunc='LReLU', use_bn=True):
    block = []
    if up == 'deconv':
        block += [deconv_layer(cin, cout, afunc=afunc, use_bn=use_bn)]
    elif up == 'bilinear':
        block += [upconv_layer(cin, cout, afunc=afunc, mode='bilinear', use_bn=use_bn)]
    elif up == 'nearest':
        block += [upconv_layer(cin, cout, afunc=afunc, mode='nearest', use_bn=use_bn)]
    else:
        raise Exception('Unknown upsample mode: %s' % up)
    for i in range(1, layers):
        block += [conv_layer(cout, cout, k=3, stride=1, pad=1, afunc=afunc, use_bn=use_bn)]
    return nn.Sequential(*block)

# For ResNet
def make_layer(block, cin, channel, layer_num, use_bn=True): # TODO: clean
    layers = []
    for i in range(0, layer_num):
        layers.append(block(cin, channel, stride=1, downsample=None, use_bn=use_bn))
    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride)
        if use_bn:
            self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if use_bn:
            self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

def merge_feats(ws, feats):
    assert(len(ws) == len(feats))
    w_sum = torch.stack(ws, 1).sum(1)
    ws = [w / (w_sum + 1e-8) for w in ws]

    feat = ws[0] * feats[0]
    for i in range(1, len(ws)):
        feat += ws[i] * feats[i]
    return feat, ws

## HDR output processing
def merge_hdr(ws, hdrs):
    assert(len(ws) == len(hdrs))
    w_sum = torch.stack(ws, 1).sum(1)
    ws = [w / (w_sum + 1e-8) for w in ws]

    hdr = ws[0] * hdrs[0]
    for i in range(1, len(ws)):
        hdr += ws[i] * hdrs[i]
    return hdr, ws

class MergeHDRModule(nn.Module):
    def __init__(self):
        super(MergeHDRModule, self).__init__()

    def forward(self, ws, hdrs):
        assert(len(ws) == len(hdrs))
        w_sum = torch.stack(ws, 1).sum(1)
        ws = [w / (w_sum + 1e-8) for w in ws]

        hdr = ws[0] * hdrs[0]
        for i in range(1, len(ws)):
            hdr += ws[i] * hdrs[i]
        return hdr, ws

def coords_grid(b, h, w, device):
    coords = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(b, 1, 1, 1)

def backward_warp(img, flow, pad='zeros'):
    b, c, h, w = img.shape
    grid = coords_grid(b, h, w, device=img.device)
    grid = grid + flow
    xgrid, ygrid = grid.split([1,1], dim=1)
    xgrid = 2*xgrid/(w-1) - 1
    ygrid = 2*ygrid/(h-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=1)
    warped_img = F.grid_sample(input=img, grid=grid.permute(0,2,3,1), mode='bilinear',  padding_mode='zeros') 
    return warped_img

def affine_warp(img, theta): # warp img1 to img2
    n, c, h, w = img.shape
    affine_grid = F.affine_grid(theta, img.shape)
    invalid_mask = ((affine_grid.narrow(3, 0, 1).abs() > 1) + (affine_grid.narrow(3, 1, 1).abs() > 1)) >= 1
    invalid_mask = invalid_mask.view(n, 1, h, w).float()
    img1_to_img2 = F.grid_sample(img, affine_grid)
    img1_to_img2 = img * invalid_mask + img1_to_img2 * (1 - invalid_mask)

    return img1_to_img2

def ldr_to_hdr(ldr, expo, gamma=2.2):
    ldr = ldr.clamp(0, 1)
    ldr = torch.pow(ldr, gamma)
    expo = expo.view(-1, 1, 1, 1)
    hdr = ldr / expo
    return hdr

def adj_expo_ldr_to_ldr(ldr, c_exp, p_exp, gamma=2.2):
    gain = torch.pow(p_exp / c_exp, 1.0 / gamma)
    gain = gain.view(-1, 1, 1, 1)
    ldr = (ldr * gain).clamp(0, 1)
    return ldr

def resize_flow(flow, h, w):
    b, c, c_h, c_w = flow.shape
    flow = F.interpolate(input=flow, size=(h, w), mode='bilinear', align_corners=False)
    flow[:, 0] *= float(w) / float(c_w) # rescale flow magnitude after rescaling spatial size
    flow[:, 1] *= float(h) / float(c_h)
    return flow

def resize_img_to_factor_of_k(img, k=64, mode='bilinear'):
    b, c, h, w = img.shape
    new_h = int(np.ceil(h / float(k)) * k)
    new_w = int(np.ceil(w / float(k)) * k)
    img = F.interpolate(input=img, size=(new_h, new_w), mode=mode, align_corners=False)
    return img

def pad_img_to_factor_of_k(img, k=64, mode='replicate'):
    if img.ndimension() == 4:
        b, c, h, w = img.shape
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')
    elif img.ndimension() == 5:
        h, w = img.shape[-2], img.shape[-1]
        pad_h, pad_w = k - h % k, k - w % k
        img = F.pad(img, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')
    return img

