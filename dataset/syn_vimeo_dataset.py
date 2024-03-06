"""
Dataloader for processing vimeo videos into training data
"""
import os
import numpy as np
from imageio import imread

import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
from utils.utils import *

np.random.seed(0)

class Syn_Vimeo_Dataset(Dataset):

    def __init__(self, root_dir, nframes, nexps, is_training=True):

        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps

        if is_training:
            list_name = 'sep_trainlist.txt'
            self.repeat = 1
        else:
            list_name = 'sep_testlist.txt'
            self.repeat = 1

        self.patch_list = read_list(os.path.join(self.root_dir, list_name))

        if not is_training:
            self.patch_list = self.patch_list[:100] # only use 100 val patches

    def __getitem__(self, index):

        img_dir = os.path.join(self.root_dir, 'sequences', self.patch_list[index // self.repeat])
        img_idxs = sorted(np.random.permutation(7)[:self.nframes] + 1)
        if np.random.random() > 0.5: # inverse time order
            img_idxs = img_idxs[::-1]
        img_paths = [os.path.join(img_dir, 'im%d.png' % idx) for idx in img_idxs]

        if self.nexps == 2:
            exposures = self._get_2exposures(index)
        elif self.nexps == 3:
            exposures = self._get_3exposures(index)
        else:
            raise Exception("Unknow exposures")

        """ sample parameters for the camera curves"""
        n, sigma = self.sample_camera_curve() # print(n, sigma)
        
        hdrs = []
        for img_path in img_paths:
            img = (imread(img_path).astype(np.float32) / 255.0).clip(0, 1)

            """ convert the LDR images to linear HDR image"""
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        h, w, c = hdrs[0].shape
        crop_h, crop_w = 256, 256

        hdrs = random_flip_lrud(hdrs)
        hdrs = random_crop(hdrs, [crop_h, crop_w])
        color_permute = np.random.permutation(3)
        for i in range(len(hdrs)):
            hdrs[i] = hdrs[i][:,:,color_permute]
        
        hdrs, ldrs = self.re_expose_ldrs(hdrs, exposures)
        ldrs_tensor = []
        hdrs_tensor = []
        expos_tensor = []
        for i in range(len(ldrs)):
            ldr = ldrs[i].astype(np.float32).transpose(2, 0, 1)
            ldr_tensor = torch.from_numpy(ldr)
            ldrs_tensor.append(ldr_tensor)

            hdr = hdrs[i].astype(np.float32).transpose(2, 0, 1)
            hdr_tensor = torch.from_numpy(hdr)
            hdrs_tensor.append(hdr_tensor)

            expos_tensor.append(torch.tensor(exposures[i]))

        flow_gts = []
        if self.nexps == 2:
            
            prev_flow = torch.zeros(2, crop_h, crop_w)
            nxt_flow = torch.zeros(2, crop_h, crop_w)
            flow_gts.append(prev_flow)
            flow_gts.append(nxt_flow)

        elif self.nexps == 3:
            prev2_flow = torch.zeros(2, crop_h, crop_w)
            nxt1_flow = torch.zeros(2, crop_h, crop_w)
            prev1_flow = torch.zeros(2, crop_h, crop_w)
            nxt2_flow = torch.zeros(2, crop_h, crop_w)

            flow_gts.append(prev2_flow)
            flow_gts.append(nxt1_flow)
            flow_gts.append(prev1_flow)
            flow_gts.append(nxt2_flow)

        else:
            raise Exception("Unknow exposures")

        flow_mask = torch.tensor(0.)

        sample = {
            'hdrs': hdrs_tensor, 
            'ldrs': ldrs_tensor,
            'expos': expos_tensor,
            'flow_gts': flow_gts,
            'flow_mask': flow_mask
            }
        return sample

    def sample_camera_curve(self):
        n = np.clip(np.random.normal(0.65, 0.1), 0.4, 0.9)
        sigma = np.clip(np.random.normal(0.6, 0.1), 0.4, 0.8)
        return n, sigma

    def apply_sigmoid_curve(self, x, n, sigma):
        y = (1 + sigma) * np.power(x, n) / (np.power(x, n) + sigma)
        return y

    def apply_inv_sigmoid_curve(self, y, n, sigma):
        x = np.power((sigma * y) / (1 + sigma - y), 1/n)
        return x

    def apply_inv_s_curve(self, y):
        x = 0.5 - np.sin(np.arcsin(1 - 2*y)/3.0)
        return x

    def discretize_to_uint16(self, img):
        max_int = 2**16-1
        img_uint16 = np.uint16(img * max_int).astype(np.float32) / max_int
        return img_uint16
    
    def _get_2exposures(self, index):
        cur_high = True if np.random.uniform() > 0.5 else False
        exposures = np.ones(self.nframes, dtype=np.float32)
        high_expo = np.random.choice([4., 8.])

        if cur_high:
            for i in range(0, self.nframes, 2):
                exposures[i] = high_expo
        else:
            for i in range(1, self.nframes, 2):
                exposures[i] = high_expo
        return exposures

    def _get_3exposures(self, index):
        if index % self.nexps == 0:
            exp1 = 1
        elif index % self.nexps == 1:
            exp1 = 4
        else:
            exp1 = 16
        expos = [exp1]
        for i in range(1, self.nframes):
            if expos[-1] == 1:
                expos.append(4)
            elif expos[-1] == 4:
                expos.append(16)
            elif expos[-1] == 16:
                expos.append(1)
            else:
                raise Exception('Unknown expos %d' % expos[-1])
        exposures = np.array(expos).astype(np.float32)
        return exposures

    def re_expose_ldrs(self, hdrs, exposures):
        mid = len(hdrs) // 2
        new_hdrs = []
        if self.nexps == 3:
            if exposures[mid] == 1:
                factor = np.random.uniform(0.1, 0.8)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # exposures[mid] == 4 or 8
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)
        else:
            if exposures[mid] == 1: # low exposure reference
                factor = np.random.uniform(0.1, 0.8)
                anchor = hdrs[mid].max()
                new_anchor = anchor * factor
            else: # high exposure reference
                percent = np.random.uniform(98, 100)
                anchor = np.percentile(hdrs[mid], percent)
                new_anchor = np.random.uniform(anchor, 1)
        for idx, hdr in enumerate(hdrs):
            new_hdr = (hdr / (anchor + 1e-8) * new_anchor).clip(0, 1)
            new_hdrs.append(new_hdr)

        ldrs = []
        for i in range(len(new_hdrs)):
            ldr = hdr_to_ldr(new_hdrs[i], exposures[i])
            ldrs.append(ldr)
        return new_hdrs, ldrs

    def __len__(self):
        return len(self.patch_list) * self.repeat