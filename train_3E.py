import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import fetch_dataloader
from models.loss import HDRFlow_Loss_3E
from models.model_3E import HDRFlow
from utils.utils import *

def get_args():
    parser = argparse.ArgumentParser(description='HDRFlow',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_vimeo_dir", type=str, default='data/vimeo_septuplet',
                        help='dataset directory'),
    parser.add_argument("--dataset_sintel_dir", type=str, default='/mnt/workspace/xugangwei/data/Sintel/training/',
                        help='dataset directory'),
    parser.add_argument('--logdir', type=str, default='./checkpoints_3E',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    # Training
    parser.add_argument('--resume', type=str, default=None,
                        help='load model from a .pth file')
    parser.add_argument('--seed', type=int, default=443, metavar='S',
                        help='random seed (default: 443)')
    parser.add_argument('--init_weights', action='store_true', default=False,
                        help='init model weights')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--lr_decay_epochs', type=str, 
                        default="20,30:2", help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='start epoch of training (default: 1)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='training batch size (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=8, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()


def train(args, model, device, train_loader, optimizer, epoch, hdrflow_loss):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for batch_idx, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        ldrs = [x.to(device) for x in batch_data['ldrs']]
        expos = [x.to(device) for x in batch_data['expos']]
        hdrs = [x.to(device) for x in batch_data['hdrs']]
        flow_gts = [x.to(device) for x in batch_data['flow_gts']]
        flow_mask = batch_data['flow_mask'].to(device)
        pred_hdr, flow_preds = model(ldrs, expos)
        cur_ldr = ldrs[2]
        loss = hdrflow_loss(pred, hdrs, flow_preds, cur_ldr, flow_mask, flow_gts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:3f})'.format(
                epoch,
                batch_idx,
                len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(),
                batch_time=batch_time,
                data_time=data_time
            ))

def validation(args, model, device, val_loader, optimizer, epoch):
    model.eval()
    n_val = len(val_loader)
    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            ldrs = [x.to(device) for x in batch_data['ldrs']]
            expos = [x.to(device) for x in batch_data['expos']]
            hdrs = [x.to(device) for x in batch_data['hdrs']]
            gt_hdr = hdrs[2]
            pred_hdr, _ = model(ldrs, expos)
            psnr = batch_psnr(pred_hdr, gt_hdr, 1.0)
            mu_psnr = batch_psnr_mu(pred_hdr, gt_hdr, 1.0)
            val_psnr.update(psnr.item())
            val_mu_psnr.update(mu_psnr.item())

    print('Validation set: Number: {}'.format(n_val))
    print('Validation set: Average PSNR-l: {:.4f}, PSNR-mu: {:.4f}'.format(val_psnr.avg, val_mu_psnr.avg))

    save_dict = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.logdir, 'checkpoint_%s.pth' % (epoch+1)))

    with open(os.path.join(args.logdir, 'checkpoint.json'), 'a') as f:
        f.write('epoch:' + str(epoch) + '\n')
        f.write('Validation set: Average PSNR-l: {:.4f}, PSNR-mu: {:.4f}\n'.format(val_psnr.avg, val_mu_psnr.avg))

def main():
    args = get_args()
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    device = torch.device('cuda')
    # model
    model = HDRFlow()
    if args.init_weights:
        init_parameters(model)
    hdrflow_loss = HDRFlow_Loss_3E().to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    model.to(device)
    model = nn.DataParallel(model)

    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("===> Loaded checkpoint: epoch {}".format(checkpoint['epoch']))
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))
    
    train_loader, val_loader = fetch_dataloader(args)

    for epoch in range(args.epochs):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, hdrflow_loss)
        validation(args, model, device, val_loader, optimizer, epoch)

if __name__ == '__main__':
    main()
