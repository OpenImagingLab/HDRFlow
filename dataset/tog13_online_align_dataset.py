import os
import numpy as np
from imageio import imread
import sys
sys.path.append('..')
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from utils.utils import *
np.random.seed(0)

class TOG13_online_align_Dataset(Dataset):
    def __init__(self, root_dir, nframes, nexps, align=True):

        self.root_dir = root_dir
        self.nframes = nframes
        self.nexps = nexps
        crf_path = root_dir.split('/')
        crf_path = crf_path[:-1]
        crf_path = '/'.join(crf_path)
        self.crf = sio.loadmat(os.path.join(crf_path, 'BaslerCRF.mat'))['BaslerCRF']
        self.align = align   
        img_list, hdr_list = self._load_img_hdr_list(self.root_dir)
        e_list = self._load_exposure_list(os.path.join(self.root_dir, 'Exposures.txt'), img_num=len(img_list))
        self.imgs_list, self.hdrs_list, self.expos_list = self._lists_to_paired_lists([img_list, hdr_list, e_list])
        print('[%s] totaling  %d ldrs' % (self.__class__.__name__, len(self.imgs_list)))
    
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

    def load_affine_matrices(self, img_path, h, w):
        dir_name, img_name = os.path.dirname(img_path), os.path.basename(img_path)
        cv2_match = np.genfromtxt(os.path.join(dir_name, 'Affine_Trans_Matrices', img_name[:-4]+'_match.txt'), dtype=np.float32)
        # For two exposure: cv2_match [2, 6], row 1->2: cur->prev, cur->next
        # For three exposure: cv2_match [4, 6], row1->4: cur->prev2, cur->prev, cur->next, cur->next2

        n_matches =cv2_match.shape[0]
        if self.nexps == 2:
            assert (n_matches == 2)
        elif self.nexps == 3:
            assert (n_matches == 4)

        cv2_match = cv2_match.reshape(n_matches, 2, 3)
        theta = np.zeros((n_matches, 2, 3)).astype(np.float32) # Theta for affine transformation in pytorch
        for mi in range(n_matches):
            theta[mi] = cvt_MToTheta(cv2_match[mi], w, h)
        return theta

    def __getitem__(self, index):
        ldrs = []
        expos = []
        matches = []
        img_paths = self.imgs_list[index]
        exposures_all = np.array(self.expos_list[index]).astype(np.float32)
        for i in range(0, self.nframes):
            img = apply_gamma(read_16bit_tif(img_paths[i], crf=self.crf), gamma=2.2)
            ldrs.append(img)
            expos.append(exposures_all[i])
            if self.align:
                match = self.load_affine_matrices(img_paths[i], img.shape[0], img.shape[1])
                matches.append(match) 
        ldrs_tensor = []
        expos_tensor = []
        matches_tensor = []
        for i in range(len(ldrs)):
            ldr = ldrs[i].astype(np.float32).transpose(2, 0, 1)
            ldr_tensor = torch.from_numpy(ldr)
            ldrs_tensor.append(ldr_tensor)
            expos_tensor.append(torch.tensor(expos[i]))
            matches_tensor.append(torch.tensor(matches[i]))

        sample = {
            'ldrs': ldrs_tensor,
            'expos': expos_tensor,
            'matches': matches_tensor
            }
        return sample

    def __len__(self):
        return len(self.imgs_list)

