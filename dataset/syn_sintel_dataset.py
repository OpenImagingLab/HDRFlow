import os
import numpy as np
# from imageio import imread
import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
from utils.utils import *
from utils import frame_utils
from imageio import imread

np.random.seed(0)

class Syn_Sintel_Dataset(Dataset):

    def __init__(self, root_dir, dtype, nframes, nexps, is_training=True):
        self.root_dir = os.path.join(root_dir, dtype)
        self.dtype = dtype
        self.nframes = nframes
        self.nexps = nexps

        if is_training:
            list_name = 'trainlist.txt'
            self.repeat = 20
        else:
            list_name = 'testlist.txt'
            self.repeat = 1

        self.scene_list = read_list(os.path.join(self.root_dir, list_name))
        image_list = []
        for scene in self.scene_list:
            files = os.listdir(os.path.join(self.root_dir, scene))
            files.sort()
            for i in range(len(files)- (self.nframes-1)):
                image = files[i:i+self.nframes]
                if all(idx.endswith('.png') for idx in image):
                    image_list.append([os.path.join(self.root_dir, scene, ip) for ip in image])

        self.image_list = image_list

    def __getitem__(self, index):

        img_paths = self.image_list[index//self.repeat]
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
            # img = read_16bit_tif(img_path)
            """ convert the LDR images to linear HDR image"""
            linear_img = self.apply_inv_sigmoid_curve(img, n, sigma)
            linear_img = self.discretize_to_uint16(linear_img)
            hdrs.append(linear_img)

        # p_hdr, c_hdr, n_hdr = hdrs[0], hdrs[1], hdrs[2]
        ### flow
        if self.nexps == 2:
            prev_flow_path = img_paths[1].replace(self.dtype, 'reverse_flow').replace('png', 'flo')
            nxt_flow_path = img_paths[1].replace(self.dtype, 'flow').replace('png', 'flo')
            prev_flow = frame_utils.read_gen(prev_flow_path)
            nxt_flow = frame_utils.read_gen(nxt_flow_path)
            flow_gts = [prev_flow, nxt_flow]
           
        elif self.nexps == 3:
            prev2_flow_path = img_paths[2].replace(self.dtype, 'reverse_flow_2').replace('png', 'flo')
            nxt1_flow_path = img_paths[2].replace(self.dtype, 'flow').replace('png', 'flo')
            prev1_flow_path = img_paths[2].replace(self.dtype, 'reverse_flow').replace('png', 'flo')
            nxt2_flow_path = img_paths[2].replace(self.dtype, 'flow_2').replace('png', 'flo')

            prev2_flow = frame_utils.read_gen(prev2_flow_path)
            nxt1_flow = frame_utils.read_gen(nxt1_flow_path)
            prev1_flow = frame_utils.read_gen(prev1_flow_path)
            nxt2_flow = frame_utils.read_gen(nxt2_flow_path)

            flow_gts = [prev2_flow, nxt1_flow, prev1_flow, nxt2_flow]

        else:
            raise Exception("Unknow exposures")

        # hdrs = prob_center_crop(hdrs)
        h, w, c = hdrs[0].shape
        # crop_h, crop_w = self.args.crop_h, self.args.crop_w

        crop_h, crop_w = 256, 256

        _hdrs = []
        _flow_gts = []

        ### random flip
        if np.random.rand() < 0.3: # h-flip
            for hdr in hdrs:
                _hdrs.append(hdr[:, ::-1])
            for flow in flow_gts:
                _flow_gts.append(flow[:, ::-1] * [-1.0, 1.0])

            hdrs = _hdrs
            flow_gts = _flow_gts
        
        
        _hdrs = []
        _flow_gts = []

        if np.random.rand() < 0.1: # v-flip
            
            for hdr in hdrs:
                _hdrs.append(hdr[::-1, :])
            for flow in flow_gts:
                _flow_gts.append(flow[::-1, :] * [1.0, -1.0])

            hdrs = _hdrs
            flow_gts = _flow_gts

        ### random crop
        y0 = np.random.randint(0, hdrs[0].shape[0] - crop_h)
        x0 = np.random.randint(0, hdrs[0].shape[1] - crop_w)

        _hdrs = []
        _flow_gts = []

        for hdr in hdrs:
            _hdrs.append(hdr[y0:y0+crop_h, x0:x0+crop_w])
        for flow in flow_gts:
            _flow_gts.append(flow[y0:y0+crop_h, x0:x0+crop_w])
        
        hdrs = _hdrs
        flow_gts = _flow_gts

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
        
        flow_gts_tensor = []
        for i in range(len(flow_gts)):
            flow = flow_gts[i].astype(np.float32).transpose(2, 0, 1)
            flow_tensor = torch.from_numpy(flow)
            flow_gts_tensor.append(flow_tensor)

        flow_mask = torch.tensor(1.)   

        sample = {
            'hdrs': hdrs_tensor, 
            'ldrs': ldrs_tensor,
            'expos': expos_tensor,
            'flow_gts': flow_gts_tensor,
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
        return len(self.image_list) * self.repeat