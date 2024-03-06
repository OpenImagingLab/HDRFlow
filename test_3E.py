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
from dataset.real_benchmark_dataset import Real_Benchmark_Dataset
from dataset.syn_test_dataset import Syn_Test_Dataset
from models.model_3E import HDRFlow
from utils.utils import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import flow_viz

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset", type=str, default='DeepHDRVideo', choices=['DeepHDRVideo', 'CinematicVideo'],
                        help='dataset directory')
parser.add_argument("--dataset_dir", type=str, default='./data/dynamic_RGB_data_3exp_release',
                        help='dataset directory')
# parser.add_argument("--dataset_dir", type=str, default='./data/static_RGB_data_3exp_rand_motion_release',
#                         help='dataset directory')
# parser.add_argument("--dataset_dir", type=str, default='./data/HDR_Synthetic_Test_Dataset',
#                         help='dataset directory')
parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/3E/checkpoint.pth')
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./output_results/3E")

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

    # test_loader = fetch_benchmarkloader(args)
    if args.dataset == 'DeepHDRVideo':
        test_dataset = Real_Benchmark_Dataset(root_dir=args.dataset_dir, nframes=5, nexps=3)   
    elif args.dataset == 'CinematicVideo':
        test_dataset = Syn_Test_Dataset(root_dir=args.dataset_dir, nframes=5, nexps=3)
    # elif args.dataset == 'HDRVideo_TOG13':
    #     test_dataset = TOG13_online_align_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2)
    else:
        print('Unknown dataset')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()

    low_psnr_l = AverageMeter()
    low_psnr_mu = AverageMeter()
    mid_psnr_l = AverageMeter()
    mid_psnr_mu = AverageMeter()
    high_psnr_l = AverageMeter()
    high_psnr_mu = AverageMeter()

    with torch.no_grad():
        for idx, img_data in enumerate(test_loader):
            ldrs = [x.to(device) for x in img_data['ldrs']]
            expos = [x.to(device) for x in img_data['expos']]
            # hdrs = [x.to(device) for x in img_data['hdrs']]
            gt_hdr = img_data['hdr']

            padder = InputPadder(ldrs[0].shape, divis_by=16)
            pad_ldrs = padder.pad(ldrs)
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
                mask = (Y < 0.2) | (Y > 0.8)
            else:
                mask = Y > 0.8
            cur_linear_ldr = ldr_to_hdr(ldrs[2], expos[2])
            cur_linear_ldr = torch.squeeze(cur_linear_ldr.cpu()).numpy().astype(np.float32).transpose(1,2,0)
            pred_hdr = (~mask) * cur_linear_ldr + (mask) * pred_hdr
            gt_hdr = torch.squeeze(gt_hdr).numpy().astype(np.float32).transpose(1,2,0)
            pred_hdr = pred_hdr.copy()
            psnrL = psnr(gt_hdr, pred_hdr)

            gt_hdr_tm = tonemap(gt_hdr)
            pred_hdr_tm = tonemap(pred_hdr)
            psnrT = psnr(gt_hdr_tm, pred_hdr_tm)

            ssimL = ssim(gt_hdr, pred_hdr, multichannel=True, channel_axis=2, data_range=gt_hdr.max()-gt_hdr.min())
            ssimT = ssim(gt_hdr_tm, pred_hdr_tm, multichannel=True, channel_axis=2, data_range=gt_hdr_tm.max()-gt_hdr_tm.min())

            psnr_l.update(psnrL)
            ssim_l.update(ssimL)
            psnr_mu.update(psnrT)
            ssim_mu.update(ssimT)

            if expos[2] == 1.:
                print("Iter: [{}/{}] Low Exposure".format(idx+1, len(test_loader)))
                low_psnr_l.update(psnrL)
                low_psnr_mu.update(psnrT)
            
            elif expos[2] == 4.:
                print("Iter: [{}/{}] Mid Exposure".format(idx+1, len(test_loader)))
                mid_psnr_l.update(psnrL)
                mid_psnr_mu.update(psnrT)

            else:
                print("Iter: [{}/{}] High Exposure".format(idx+1, len(test_loader)))
                high_psnr_l.update(psnrL)
                high_psnr_mu.update(psnrT)
            
            print("PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnrT, psnrL))
            print("SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssimT, ssimL))

            # save results
            if args.save_results:
                dataset_name = args.dataset_dir.split('/')[-1]
                hdr_output_dir = os.path.join(args.save_dir, dataset_name, 'hdr_output')
                if not osp.exists(hdr_output_dir):
                    os.makedirs(hdr_output_dir)
                cv2.imwrite(os.path.join(hdr_output_dir, '{}_pred.png'.format(idx+1)), (pred_hdr_tm*255.)[:,:,[2,1,0]].astype('uint8'))
                # save_hdr(os.path.join(args.save_dir, img_data['hdr_path'][0]), pred_hdr)
    print("Low Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(low_psnr_mu.avg, low_psnr_l.avg))
    print("Mid Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(mid_psnr_mu.avg, mid_psnr_l.avg))
    print("High Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(high_psnr_mu.avg, high_psnr_l.avg))
    print("Average PSNR_mu: {:.4f}  PSNR_l: {:.4f}".format(psnr_mu.avg, psnr_l.avg))
    print("Average SSIM_mu: {:.4f}  SSIM_l: {:.4f}".format(ssim_mu.avg, ssim_l.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()




