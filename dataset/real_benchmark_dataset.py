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

class Real_Benchmark_Dataset(Dataset):
    def __init__(self, root_dir, nframes, nexps):

        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps       
        self.scene_list = read_list(os.path.join(self.root_dir, 'scene_all.txt'))

        self.expos_list = []
        self.img_list = []
        self.hdrs_list = []

        for i in range(len(self.scene_list)):
            img_dir = os.path.join(self.root_dir, self.scene_list[i])
            img_list, hdr_list = self._load_img_hdr_list(img_dir)
            e_list = self._load_exposure_list(os.path.join(img_dir, 'Exposures.txt'), img_num=len(img_list))

            for i in range(0, len(img_list)-(self.nframes+2)+1):
                sub_img_list = img_list[i:i+self.nframes+2]
                sub_hdr_list = hdr_list[i:i+self.nframes+2]
                sub_e_list = e_list[i:i+self.nframes+2]

                self.expos_list.append(sub_e_list)
                self.img_list.append(sub_img_list)
                self.hdrs_list.append(sub_hdr_list)

        print('[%s] totaling  %d ldrs' % (self.__class__.__name__, len(self.img_list)))

    def _load_img_hdr_list(self, img_dir):
        if os.path.exists(os.path.join(img_dir, 'img_hdr_list.txt')):
            img_hdr_list = np.genfromtxt(os.path.join(img_dir, 'img_hdr_list.txt'), dtype='str')
            img_list = img_hdr_list[:, 0]
            hdr_list = img_hdr_list[:, 1]
        else:
            img_list = np.genfromtxt(os.path.join(img_dir, 'img_list.txt'), dtype='str')
            hdr_list = ['None'] * len(img_list)
        img_list =[os.path.join(img_dir, img_path) for img_path in img_list]
        hdr_list =[os.path.join(img_dir, hdr_path) for hdr_path in hdr_list]
        return img_list, hdr_list

    def _load_exposure_list(self, expos_path, img_num):
        expos = np.genfromtxt(expos_path, dtype='float')
        expos = np.power(2, expos - expos.min()).astype(np.float32)
        expo_list = np.tile(expos, int(img_num / len(expos) + 1))[:img_num]
        return expo_list

    def __getitem__(self, index):

        ldrs = []
        expos = []

        if self.nexps == 2:
            img_paths, hdr_path = self.img_list[index], self.hdrs_list[index][2]
            exposures_all = np.array(self.expos_list[index]).astype(np.float32)
            for i in range(1, 4):
                if img_paths[i][-4:] == '.tif':
                    img = read_16bit_tif(img_paths[i])
                else:
                    img = imread(img_paths[i]) / 255.0
                ldrs.append(img)
                expos.append(exposures_all[i])

        elif self.nexps == 3:
            img_paths, hdr_path = self.img_list[index], self.hdrs_list[index][3]
            exposures_all = np.array(self.expos_list[index]).astype(np.float32)
            for i in range(1, 6):
                if img_paths[i][-4:] == '.tif':
                    img = read_16bit_tif(img_paths[i])
                else:
                    img = imread(img_paths[i]) / 255.0
                ldrs.append(img)
                expos.append(exposures_all[i])

        else:
            raise Exception("Unknow exposures")

        if os.path.exists(hdr_path):
            hdr = read_hdr(hdr_path)
            if hdr.max() > 1:
                hdr = hdr / hdr.max()
        hdr = hdr.astype(np.float32).transpose(2, 0, 1)
        hdr_tensor = torch.from_numpy(hdr)
        ldrs_tensor = []
        expos_tensor = []
        for i in range(len(ldrs)):
            ldr = ldrs[i].astype(np.float32).transpose(2, 0, 1)
            ldr_tensor = torch.from_numpy(ldr)
            ldrs_tensor.append(ldr_tensor)
            expos_tensor.append(torch.tensor(expos[i]))       

        sample = {
            'hdr_path': hdr_path.split('/')[-2]+'_'+hdr_path.split('/')[-1],
            'hdr': hdr_tensor, 
            'ldrs': ldrs_tensor,
            'expos': expos_tensor
            }
        return sample

    def __len__(self):
        return len(self.img_list)