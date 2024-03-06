import math
import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from .network_utils import backward_warp

def tonemap(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

class HDRFlow_Loss_2E(nn.Module):
    def __init__(self, mu=5000):
        super(HDRFlow_Loss_2E, self).__init__()
        self.mu = mu

    def forward(self, pred, hdrs, flow_preds, cur_ldr, flow_mask, flow_gts):
        gt = hdrs[1]
        mu_pred = tonemap(pred, self.mu)
        mu_gt = tonemap(gt, self.mu)
        recon_loss = nn.L1Loss()(mu_pred, mu_gt)
        loss = recon_loss

        Y = 0.299 * cur_ldr[:, 0] + 0.587 * cur_ldr[:, 1] + 0.114 * cur_ldr[:, 2]
        Y = Y[:, None]
        mask = (Y>0.8) | (Y<0.2)
        mask = mask.repeat(1, 3, 1, 1)
        p_flow, n_flow = flow_preds
        if mask.sum() > 0:
            p_warp_hdr = backward_warp(hdrs[0], p_flow)
            n_warp_hdr = backward_warp(hdrs[2], n_flow)

            mu_p_warp_hdr = tonemap(p_warp_hdr, self.mu)
            mu_n_warp_hdr = tonemap(n_warp_hdr, self.mu)

            p_align_loss = nn.L1Loss()(mu_p_warp_hdr[mask], mu_gt[mask])
            n_align_loss = nn.L1Loss()(mu_n_warp_hdr[mask], mu_gt[mask])

            hdr_align_loss = p_align_loss + n_align_loss
            loss += 0.5 * hdr_align_loss

        b, c, h, w = p_flow.shape       
        flow_mask = flow_mask[:, None, None, None].repeat(1, 2, h, w)
        flow_mask = flow_mask > 0.5        
        if flow_mask.sum() > 0:
            p_flow_loss = nn.L1Loss()(p_flow[flow_mask], flow_gts[0][flow_mask])
            n_flow_loss = nn.L1Loss()(n_flow[flow_mask], flow_gts[1][flow_mask])

            flow_loss = p_flow_loss + n_flow_loss
            loss += 0.001 * flow_loss

        return loss

class HDRFlow_Loss_3E(nn.Module):
    def __init__(self, mu=5000):
        super(HDRFlow_Loss_3E, self).__init__()
        self.mu = mu

    def forward(self, pred, hdrs, flow_preds, cur_ldr, flow_mask, flow_gts):
        gt = hdrs[2]
        mu_pred = tonemap(pred, self.mu)
        mu_gt = tonemap(gt, self.mu)
        recon_loss = nn.L1Loss()(mu_pred, mu_gt)
        loss = recon_loss

        Y = 0.299 * cur_ldr[:, 0] + 0.587 * cur_ldr[:, 1] + 0.114 * cur_ldr[:, 2]
        Y = Y[:, None]
        mask = (Y>0.8) | (Y<0.2)
        mask = mask.repeat(1, 3, 1, 1)
        p2_flow, n1_flow, p1_flow, n2_flow = flow_preds
        if mask.sum() > 0:
            p2_warp_hdr = backward_warp(hdrs[0], p2_flow)
            n1_warp_hdr = backward_warp(hdrs[3], n1_flow)
            p1_warp_hdr = backward_warp(hdrs[1], p1_flow)
            n2_warp_hdr = backward_warp(hdrs[4], n2_flow)

            mu_p2_warp_hdr = tonemap(p2_warp_hdr, self.mu)
            mu_n1_warp_hdr = tonemap(n1_warp_hdr, self.mu)
            mu_p1_warp_hdr = tonemap(p1_warp_hdr, self.mu)
            mu_n2_warp_hdr = tonemap(n2_warp_hdr, self.mu)

            p2_align_loss = nn.L1Loss()(mu_p2_warp_hdr[mask], mu_gt[mask])
            n1_align_loss = nn.L1Loss()(mu_n1_warp_hdr[mask], mu_gt[mask])
            p1_align_loss = nn.L1Loss()(mu_p1_warp_hdr[mask], mu_gt[mask])
            n2_align_loss = nn.L1Loss()(mu_n2_warp_hdr[mask], mu_gt[mask])

            hdr_align_loss = p2_align_loss + n1_align_loss + p1_align_loss + n2_align_loss
            loss += 0.2 * hdr_align_loss
        
        b, c, h, w = p2_flow.shape       
        flow_mask = flow_mask[:, None, None, None].repeat(1, 2, h, w)
        flow_mask = flow_mask > 0.5        
        if flow_mask.sum() > 0:
            p2_flow_loss = nn.L1Loss()(p2_flow[flow_mask], flow_gts[0][flow_mask])
            n1_flow_loss = nn.L1Loss()(n1_flow[flow_mask], flow_gts[1][flow_mask])
            p1_flow_loss = nn.L1Loss()(p1_flow[flow_mask], flow_gts[2][flow_mask])
            n2_flow_loss = nn.L1Loss()(n2_flow[flow_mask], flow_gts[3][flow_mask])

            flow_loss = p2_flow_loss + n1_flow_loss + p1_flow_loss + n2_flow_loss
            loss += 0.0005 * flow_loss

        return loss
