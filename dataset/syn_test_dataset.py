import os
import numpy as np
from imageio import imread
import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
from utils.utils import *
np.random.seed(0)

class Syn_Test_Dataset(Dataset):
    def __init__(self, root_dir, nframes, nexps):

        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps       
        self.scene_list = read_list(os.path.join(self.root_dir, 'scenes_2expo.txt'))

        self.expos_list = []
        self.img_list = []
        self.hdrs_list = []

        for i in range(len(self.scene_list)):
            img_dir = os.path.join(self.root_dir, 'Images', self.scene_list[i])
            img_list, hdr_list = self._load_img_hdr_list(img_dir)
            e_list = self._load_exposure_list(os.path.join(img_dir, 'Exposures.txt'), img_num=len(img_list))
            img_list, hdr_list, e_list = self._lists_to_paired_lists([img_list, hdr_list, e_list])

            self.expos_list += e_list
            self.img_list += img_list
            self.hdrs_list += hdr_list

        print('[%s] totaling  %d ldrs' % (self.__class__.__name__, len(self.img_list)))

    def _load_img_hdr_list(self, img_dir):
        scene_list = np.genfromtxt(os.path.join(img_dir, 'img_list.txt'), dtype='str')
        img_list = ['%s.tif' % name for name in scene_list]
        hdr_list = ['%s.hdr' % name for name in scene_list]
        img_list =[os.path.join(img_dir, img_path) for img_path in img_list]
        hdr_list =[os.path.join(img_dir, hdr_path) for hdr_path in hdr_list]
        return img_list, hdr_list

    def _load_exposure_list(self, expos_path, img_num):
        expos = np.genfromtxt(expos_path, dtype='float')
        expos = np.power(2, expos - expos.min()).astype(np.float32)
        expo_list = np.tile(expos, int(img_num / len(expos) + 1))[:img_num]
        return expo_list
    
    def _lists_to_paired_lists(self, lists):
        paired_lists = []

        for l in lists:
            if (self.nexps == 2 and self.nframes == 3) or (self.nexps == 3 and self.nframes == 5):
                l = l[1:-1]
            paired_list = []
            paired_list.append(l[: len(l) - self.nframes + 1])
            for j in range(1, self.nframes):
                start_idx, end_idx = j, len(l) - self.nframes + 1 + j
                paired_list.append(l[start_idx: end_idx])
            paired_lists.append(np.stack(paired_list, 1).tolist()) # Nxframes
        return paired_lists

    def __getitem__(self, index):

        ldrs = []
        expos = []
        if self.nexps == 2:
            img_paths, hdr_path = self.img_list[index], self.hdrs_list[index][1]
            exposures_all = np.array(self.expos_list[index]).astype(np.float32)
            for i in range(0, 3):
                img = read_16bit_tif(img_paths[i])
                ldrs.append(img)
                expos.append(exposures_all[i])
        
        elif self.nexps == 3:
            img_paths, hdr_path = self.img_list[index], self.hdrs_list[index][2]
            exposures_all = np.array(self.expos_list[index]).astype(np.float32)
            for i in range(0, 5):
                img = read_16bit_tif(img_paths[i])
                ldrs.append(img)
                expos.append(exposures_all[i])
        
        else:
            raise Exception("Unknow exposures")

        if os.path.exists(hdr_path):
            hdr = read_hdr(hdr_path)
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
            'hdr_path': hdr_path.split('/')[-1],
            'hdr': hdr_tensor, 
            'ldrs': ldrs_tensor,
            'expos': expos_tensor
            }
        return sample

    def __len__(self):
        return len(self.img_list)

