import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp
import sys
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
from models.model_2E import HDRFlow
from utils.utils import *
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import flow_viz

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset", type=str, default='DeepHDRVideo', choices=['DeepHDRVideo', 'CinematicVideo'],
                        help='dataset directory')
parser.add_argument("--dataset_dir", type=str, default='data/dynamic_RGB_data_2exp_release',
                        help='dataset directory')
# parser.add_argument("--dataset_dir", type=str, default='data/static_RGB_data_2exp_rand_motion_release',
#                         help='dataset directory')
# parser.add_argument("--dataset_dir", type=str, default='data/HDR_Synthetic_Test_Dataset',
#                         help='dataset directory')
parser.add_argument('--pretrained_model', type=str, default='./pretrained_models/2E/checkpoint.pth')
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./output_results/2E")

def save_flo(flow_preds, i, args):
    p_flo, n_flo = flow_preds
    p_flo = torch.squeeze(p_flo).permute(1,2,0).cpu().numpy()
    p_flo = flow_viz.flow_to_image(p_flo)
    n_flo = torch.squeeze(n_flo).permute(1,2,0).cpu().numpy()
    n_flo = flow_viz.flow_to_image(n_flo)

    concat_flo = np.concatenate([p_flo, n_flo], axis=1)
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
        test_dataset = Real_Benchmark_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2)   
    elif args.dataset == 'CinematicVideo':
        test_dataset = Syn_Test_Dataset(root_dir=args.dataset_dir, nframes=3, nexps=2)
    else:
        print('Unknown dataset')

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)


    psnrL_list = AverageMeter()
    ssimL_list = AverageMeter()
    psnrT_list = AverageMeter()
    ssimT_list = AverageMeter()

    low_psnrL_list = AverageMeter()
    low_psnrT_list = AverageMeter()
    high_psnrL_list = AverageMeter()
    high_psnrT_list = AverageMeter()

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
            cur_ldr = torch.squeeze(ldrs[1].cpu()).numpy().astype(np.float32).transpose(1,2,0)
            Y = 0.299 * cur_ldr[:, :, 0] + 0.587 * cur_ldr[:, :, 1] + 0.114 * cur_ldr[:, :, 2]
            Y = Y[:, :, None]
            if expos[1] <= 1.:
                mask = Y < 0.2
            else:
                mask = Y > 0.8
            cur_linear_ldr = ldr_to_hdr(ldrs[1], expos[1])
            cur_linear_ldr = torch.squeeze(cur_linear_ldr.cpu()).numpy().astype(np.float32).transpose(1,2,0)
            pred_hdr = (~mask) * cur_linear_ldr + (mask) * pred_hdr
            gt_hdr = torch.squeeze(gt_hdr).numpy().astype(np.float32).transpose(1,2,0)
            gt_hdr_tm = tonemap(gt_hdr)
            pred_hdr_tm = tonemap(pred_hdr)

            psnrL = psnr(gt_hdr, pred_hdr)
            ssimL = ssim(gt_hdr, pred_hdr, multichannel=True, channel_axis=2, data_range=gt_hdr.max()-gt_hdr.min())

            psnrT = psnr(gt_hdr_tm, pred_hdr_tm)
            ssimT = ssim(gt_hdr_tm, pred_hdr_tm, multichannel=True, channel_axis=2, data_range=gt_hdr_tm.max()-gt_hdr_tm.min())

            psnrL_list.update(psnrL)
            ssimL_list.update(ssimL)
            psnrT_list.update(psnrT)
            ssimT_list.update(ssimT)

            if expos[1] <= 1.:
                print("Iter: [{}/{}] Low Exposure".format(idx+1, len(test_loader)))
                low_psnrL_list.update(psnrL)
                low_psnrT_list.update(psnrT)
            else:
                print("Iter: [{}/{}] High Exposure".format(idx+1, len(test_loader)))
                high_psnrL_list.update(psnrL)
                high_psnrT_list.update(psnrT)
            
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

    print("Low Average PSNRT: {:.4f}  PSNRL: {:.4f}".format(low_psnrT_list.avg, low_psnrL_list.avg))
    print("High Average PSNRT: {:.4f}  PSNRL: {:.4f}".format(high_psnrT_list.avg, high_psnrL_list.avg))
    print("Average PSNRT: {:.4f}  PSNRL: {:.4f}".format(psnrT_list.avg, psnrL_list.avg))
    print("Average SSIMT: {:.4f}  SSIML: {:.4f}".format(ssimT_list.avg, ssimL_list.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")


if __name__ == '__main__':
    main()
