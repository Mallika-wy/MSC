"""Joint dynamics residuals in physical units, with history and adjacency masks."""
import torch
from torch.nn import functional as F


def joint_dynamics(joints, history, mask):
    if joints.shape[1] == 0 or history.shape[1] < 2:
        raise ValueError('Dynamics need a future and two history frames')
    joints = joints.float().masked_fill(~mask[...,None],0)
    full = torch.cat((history[:,-2:].float(),joints),1)
    valid = torch.cat((torch.ones_like(mask[:,:1]).expand(-1,2),mask),1)
    vel = 30*(full[:,2:]-full[:,1:-1])
    acc = 900*(full[:,2:]-2*full[:,1:-1]+full[:,:-2])
    vm = valid[:,2:] & valid[:,1:-1]
    am = vm & valid[:,:-2]
    return vel,acc,vm,am


def dynamics_loss(pred, target, history, mask, velocity_scale, acceleration_scale):
    pv,pa,vm,am = joint_dynamics(pred,history,mask)
    tv,ta,_,_ = joint_dynamics(target,history,mask)
    def residual(p,t,scale,m):
        scale = scale.float().clamp_min(1e-4)
        errors = F.smooth_l1_loss((p-t)/scale,torch.zeros_like(p),reduction='none').mean(-1)
        return errors.masked_fill(~m,0).sum()/m.sum().clamp_min(1)
    return residual(pv,tv,velocity_scale,vm),residual(pa,ta,acceleration_scale,am)
