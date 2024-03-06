import torch
import torch.nn as nn
from .network_utils import *
from .flow_2E import Flow_Net
from .fusion_2E import Fusion_Net

def cur_tone_perturb(cur, test_mode, d=0.7):
    if not test_mode:
        b, c, h, w = cur.shape
        gamma_aug = torch.exp(torch.rand(b, 3, 1, 1) * 2 * d - d)
        gamma_aug = gamma_aug.to(cur.device)
        cur_aug = torch.pow(cur, 1.0 / gamma_aug)
    else:
        cur_aug = cur
    return cur_aug

def prepare_fusion_inputs(ldrs, pt_cur, expos, flows):
    prev, cur, nxt = ldrs
    p_exp, c_exp, n_exp = expos
    p_flow, n_flow = flows
    p_warp = backward_warp(prev, p_flow)
    n_warp = backward_warp(nxt, n_flow)
    p_warp_hdr = ldr_to_hdr(p_warp, p_exp)
    n_warp_hdr = ldr_to_hdr(n_warp, n_exp)
    c_hdr = ldr_to_hdr(cur, c_exp)
    p_hdr = ldr_to_hdr(prev, p_exp)
    n_hdr = ldr_to_hdr(nxt, n_exp)
    pt_c_hdr = ldr_to_hdr(pt_cur, c_exp)
    ldrs = [pt_cur, p_warp, n_warp, prev, nxt]
    hdrs = [pt_c_hdr, p_warp_hdr, n_warp_hdr, p_hdr, n_hdr]
    fusion_in = hdrs
    fusion_in += ldrs
    fusion_in = torch.cat(fusion_in, 1)
    fusion_hdrs = [c_hdr, p_warp_hdr, n_warp_hdr, p_hdr, n_hdr]
    return fusion_in, fusion_hdrs

class HDRFlow(nn.Module):
    def __init__(self):
        super(HDRFlow, self).__init__()
        self.flow_net = Flow_Net()
        self.fusion_net = Fusion_Net(c_in=30, c_out=5, c_mid=128)  
    def forward(self, ldrs, expos, test_mode=False):
        prev, cur, nxt = ldrs
        pt_cur = cur_tone_perturb(cur, test_mode)
        # flow net
        flow_preds = self.flow_net([prev, pt_cur, nxt], expos)
        fusion_in, fusion_hdrs = prepare_fusion_inputs(ldrs, pt_cur, expos, flow_preds)
        # fusion net
        pred_hdr = self.fusion_net(fusion_in, fusion_hdrs)      
        return pred_hdr, flow_preds
