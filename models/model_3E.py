import math
import time
import torch
import torch.nn as nn
from .network_utils import *
from .flow_3E import Flow_Net
from .fusion_3E import Fusion_Net

def cur_tone_perturb(cur, test_mode, d=0.7):
    if not test_mode:
        b, c, h, w = cur.shape
        gamma_aug = torch.exp(torch.rand(b, 3, 1, 1) * 2 * d - d)
        gamma_aug = gamma_aug.to(cur.device)
        cur_aug = torch.pow(cur, 1.0 / gamma_aug)
    else:
        cur_aug = cur
    return cur_aug

def prepare_fusion_inputs(ldrs, pt_c, expos, flow_preds):
    p2, p1, c, n1, n2 = ldrs
    p2_exp, p1_exp, c_exp, n1_exp, n2_exp = expos
    p2_flow, n1_flow, p1_flow, n2_flow = flow_preds

    p2_warp = backward_warp(p2, p2_flow)
    n1_warp = backward_warp(n1, n1_flow)
    p1_warp = backward_warp(p1, p1_flow)
    n2_warp = backward_warp(n2, n2_flow)

    p2_warp_hdr = ldr_to_hdr(p2_warp, p2_exp)
    n1_warp_hdr = ldr_to_hdr(n1_warp, n1_exp)
    p1_warp_hdr = ldr_to_hdr(p1_warp, p1_exp)
    n2_warp_hdr = ldr_to_hdr(n2_warp, n2_exp)
    c_hdr = ldr_to_hdr(c, c_exp)
    p2_hdr = ldr_to_hdr(p2, p2_exp)
    p1_hdr = ldr_to_hdr(p1, p1_exp)
    n1_hdr = ldr_to_hdr(n1, n1_exp)
    n2_hdr = ldr_to_hdr(n2, n2_exp)
    pt_c_hdr = ldr_to_hdr(pt_c, c_exp)

    fusion_in = [pt_c, pt_c_hdr, p2_warp, p2_warp_hdr, n1_warp, n1_warp_hdr, p1_warp, p1_warp_hdr, n2_warp, n2_warp_hdr, p2, p2_hdr, n1, n1_hdr, p1, p1_hdr, n2, n2_hdr]
    fusion_hdrs = [c_hdr, p2_warp_hdr, n1_warp_hdr, p1_warp_hdr, n2_warp_hdr, p2_hdr, n1_hdr, p1_hdr, n2_hdr]
    fusion_in = torch.cat(fusion_in, dim=1)
    return fusion_in, fusion_hdrs

class HDRFlow(nn.Module):
    def __init__(self):
        super(HDRFlow, self).__init__()
        self.flow_net = Flow_Net()
        self.fusion_net = Fusion_Net(c_in=54, c_out=9, c_mid=256)
   
    def forward(self, ldrs, expos, test_mode=False):
        prev2, prev1, cur, nxt1, nxt2 = ldrs
        pt_cur = cur_tone_perturb(cur, test_mode)
        # flow net
        flow_preds = self.flow_net([prev2, prev1, pt_cur, nxt1, nxt2], expos)
        fusion_in, fusion_hdrs = prepare_fusion_inputs(ldrs, pt_cur, expos, flow_preds)
        # fusion net
        pred_hdr = self.fusion_net(fusion_in, fusion_hdrs)
        return pred_hdr, flow_preds
