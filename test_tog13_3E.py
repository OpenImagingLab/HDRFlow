import os
import os.path as osp
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from tqdm import tqdm
import glob
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.tog13_online_align_dataset import TOG13_online_align_Dataset
from models.model_3E import HDRFlow
from utils.utils import *
from utils import flow_viz

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default='/mnt/workspace/xugangwei/data/TOG13_Dynamic_Dataset/Dog-3Exp-2Stop',
                        help='dataset directory')
parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/3E/checkpoint.pth')
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="output_results/3E/")

def save_flo(flow_preds, i, args):
    p2_flo, n1_flo, p1_flo, n2_flo = flow_preds
    p1_flo = torch.squeeze(p1_flo).permute(1,2,0).cpu().numpy()
    p1_flo = flow_viz.flow_to_image(p1_flo)
    n1_flo = torch.squeeze(n1_flo).permute(1,2,0).cpu().numpy()
    n1_flo = flow_viz.flow_to_image(n1_flo)
    p2_flo = torch.squeeze(p2_flo).permute(1,2,0).cpu().numpy()
    p2_flo = flow_viz.flow_to_image(p2_flo)
    n2_flo = torch.squeeze(n2_flo).permute(1,2,0).cpu().numpy()
    n2_flo = flow_viz.flow_to_image(n2_flo)
    concat_flo = np.concatenate([p2_flo, n1_flo, p1_flo, n2_flo], axis=1)
    dataset_name = args.dataset_dir.split('/')[-1]
    save_dir = os.path.join(args.save_dir, dataset_name, 'flow_preds')
    os.makedirs(save_dir, exist_ok=True)
    flo_path = os.path.join(save_dir, f'{i}_flow.png')
    cv2.imwrite(flo_path, concat_flo[:, :, [2,1,0]].astype('uint8'))

def main():
    # Settings
    args = parser.parse_args()
    # pretrained_model
    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)
    device = torch.device('cuda')
    model = HDRFlow()
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
    model.eval()
    test_dataset = TOG13_online_align_Dataset(root_dir=args.dataset_dir, nframes=5, nexps=3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    with torch.no_grad():
        for idx, img_data in enumerate(test_loader):
            ldrs = [x.to(device) for x in img_data['ldrs']]
            expos = [x.to(device) for x in img_data['expos']]
            matches = [x.to(device) for x in img_data['matches']]
            # hdrs = [x.to(device) for x in img_data['hdrs']]
            align_ldrs = global_align_nbr_ldrs(ldrs, matches)
            padder = InputPadder(align_ldrs[0].shape, divis_by=16)
            pad_ldrs = padder.pad(align_ldrs)
            pred_hdr, flow_preds = model(pad_ldrs, expos, test_mode=True)           
            pred_hdr = padder.unpad(pred_hdr)
            pred_hdr = torch.squeeze(pred_hdr.detach().cpu()).numpy().astype(np.float32).transpose(1,2,0)
            save_flo(flow_preds, idx+1, args)
            cur_ldr = torch.squeeze(ldrs[2].cpu()).numpy().astype(np.float32).transpose(1,2,0)
            Y = 0.299 * cur_ldr[:, :, 0] + 0.587 * cur_ldr[:, :, 1] + 0.114 * cur_ldr[:, :, 2]
            Y = Y[:, :, None]
            if expos[2] == 1.:
                mask = Y < 0.2
            elif expos[2] == 4.:
                mask = (Y<0.2) | (Y>0.8)
            else:
                mask = Y > 0.8

            cur_linear_ldr = ldr_to_hdr(ldrs[2], expos[2])
            cur_linear_ldr = torch.squeeze(cur_linear_ldr.cpu()).numpy().astype(np.float32).transpose(1,2,0)
            pred_hdr = (~mask) * cur_linear_ldr + (mask) * pred_hdr
            # save results
            if args.save_results:
                dataset_name = args.dataset_dir.split('/')[-1]
                hdr_output_dir = os.path.join(args.save_dir, dataset_name, 'hdr_output')
                if not osp.exists(hdr_output_dir):
                    os.makedirs(hdr_output_dir)
                save_hdr(os.path.join(hdr_output_dir, '{}_pred.hdr'.format(idx+1)), pred_hdr)

if __name__ == '__main__':
    main()




