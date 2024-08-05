import numpy as np
import os, glob
import cv2
import math
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import resize, rotate
import torch.nn.functional as F

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_expo_times(file_name):
    return np.power(2, np.loadtxt(file_name))

def read_images(file_names):
    imgs = []
    for img_str in file_names:
        img = cv2.imread(img_str, -1)
        # equivalent to im2single from Matlab
        img = img / 2 ** 16
        img = np.float32(img)
        img.clip(0, 1)
        imgs.append(img)
    return np.array(imgs)

def read_label(file_path, file_name):
    label = imageio.imread(os.path.join(file_path, file_name), 'hdr')
    label = label[:, :, [2, 1, 0]]  ##cv2
    return label

def ldr_to_hdr(img, expo, gamma=2.2):
    return (img ** gamma) / (expo + 1e-8)

# def hdr_to_ldr(img, expo, gamma=2.2, stdv1=1e-3, stdv2=1e-3):
def hdr_to_ldr(img, expo, gamma=2.2, stdv1=1e-3, stdv2=1e-3):
    # add noise to low expo
    if expo == 1. or expo == 4.:
        stdv = np.random.rand(*img.shape) * (stdv2 - stdv1) + stdv1
        noise = np.random.normal(0, stdv)
        img = (img + noise).clip(0, 1)
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img

def tonemap(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)

# def psnr(x, target):
#     sqrdErr = np.mean((x - target) ** 2)
#     return 10 * log10(1/sqrdErr)

def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

# def adjust_learning_rate(args, optimizer, epoch):
#     lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(args, optimizer, epoch):
    splits = args.lr_decay_epochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    downscale_rate = float(splits[1])
    print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

    lr = args.lr
    for eid in downscale_epochs:
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2

#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())

#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()

def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def save_hdr(path, image):
    return radiance_writer(path, image)

def read_16bit_tif(img_name, crf=None):
    img = cv2.imread(img_name, -1) #/ 65535.0 # normalize to [0, 1]
    img = img[:, :, [2, 1, 0]] # BGR to RGB
    if crf is not None:
        img = reverse_crf(img, crf)
        img = img / crf.max()
    else:
        img = img / 65535.0
    return img

def prob_center_crop(imgs):
    if np.random.uniform() > 0.9:
        return imgs
    new_imgs = []
    for img in imgs:
        h, w, c = img.shape
        t, l = h // 4, w // 6
        new_imgs.append(img[t:, l:w-l])
    return new_imgs

def random_crop(inputs, size, margin=0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img in inputs:
            outputs.append(img[t: t+c_h, l: l+c_w])
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs

def random_flip_lrud(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img in inputs:
        flip_img = np.fliplr(img)
        if vertical_flip:
            flip_img = np.flipud(flip_img)
        outputs.append(flip_img.copy())
    if not is_list: outputs = outputs[0]
    return outputs

def random_rotate(inputs, angle=90.0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    ang = np.random.uniform(0, angle)
    for img in inputs:
        outputs.append(rotate(img, angle=ang, mode='constant'))
    if not is_list: outputs = outputs[0]
    return outputs

def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def read_hdr(filename, use_cv2=True):
    ext = os.path.splitext(filename)[1]
    if use_cv2:
        hdr = cv2.imread(filename, -1)[:,:,::-1].clip(0)
    elif ext == '.exr':
        hdr = read_exr(filename) 
    elif ext == '.hdr':
        hdr = cv2.imread(filename, -1)
    elif ext == '.npy':
        hdr = np.load(filenmae) 
    else:
        raise_not_defined()
    return hdr

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

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

def reverse_crf(img, crf):
    img = img.astype(int)
    out = img.astype(float)
    for i in range(img.shape[2]):
        out[:,:,i] = crf[:,i][img[:,:,i]] # crf shape [65536, 3]
    return out

# For online global alignment
def cvt_MToTheta(M, w, h):
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]

def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N

def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    # TODO: do this analytically maybe?
    N = get_N(W, H)
    return np.linalg.inv(N)

def apply_gamma(image, gamma=2.2):
    image = image.clip(1e-8, 1)
    image = np.power(image, 1.0 / gamma)
    return image

def affine_warp(img, theta): # warp img1 to img2
    n, c, h, w = img.shape
    affine_grid = F.affine_grid(theta, img.shape)
    invalid_mask = ((affine_grid.narrow(3, 0, 1).abs() > 1) + (affine_grid.narrow(3, 1, 1).abs() > 1)) >= 1
    invalid_mask = invalid_mask.view(n, 1, h, w).float()
    img1_to_img2 = F.grid_sample(img, affine_grid)
    img1_to_img2 = img * invalid_mask + img1_to_img2 * (1 - invalid_mask)
    return img1_to_img2

def global_align_nbr_ldrs(ldrs, matches):
    if len(ldrs) == 3:
        match_p = matches[0][:,1].view(-1, 2, 3)
        match_n = matches[2][:,0].view(-1, 2, 3)
        p_to_c = affine_warp(ldrs[0], match_p)
        n_to_c = affine_warp(ldrs[2], match_n)
        return [p_to_c, ldrs[1], n_to_c]

    elif len(ldrs) == 5:
        match_p2 = matches[0][:,3].view(-1, 2, 3) 
        match_p1 = matches[1][:,2].view(-1, 2, 3) 
        match_n1 = matches[3][:,1].view(-1, 2, 3)
        match_n2 = matches[4][:,0].view(-1, 2, 3)
        p2_to_c = affine_warp(ldrs[0], match_p2)
        p1_to_c = affine_warp(ldrs[1], match_p1)
        n1_to_c = affine_warp(ldrs[3], match_n1)
        n2_to_c = affine_warp(ldrs[4], match_n2)
        return [p2_to_c, p1_to_c, ldrs[2], n1_to_c, n2_to_c]
    else:
        return 0
